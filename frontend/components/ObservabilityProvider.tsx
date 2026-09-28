"use client";

import { useEffect } from "react";
import { initFrontendMonitoring } from "../lib/observability/monitoring";

/**
 * Client-side component to initialize provider-independent frontend observability
 * on browser mount. Safe no-op during SSR/static generation and when DSN is absent.
 */
export default function ObservabilityProvider() {
  useEffect(() => {
    initFrontendMonitoring();
  }, []);

  return null;
}
