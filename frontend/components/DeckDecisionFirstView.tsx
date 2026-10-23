"use client";

/**
 * frontend/components/DeckDecisionFirstView.tsx
 *
 * Backward-compatibility wrapper for DeckDecisionFirstView.
 * Delegates to DecisionFirstDossierView governed by ARX Closure Model v2.
 */

import React from "react";
import DecisionFirstDossierView, {
  DecisionFirstDossierViewProps,
} from "./DecisionFirstDossierView";

export type DeckDecisionFirstViewProps = DecisionFirstDossierViewProps;

export default function DeckDecisionFirstView(props: DecisionFirstDossierViewProps) {
  return <DecisionFirstDossierView {...props} />;
}
