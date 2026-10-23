"use client";

import React from "react";
import { ScenarioFixture } from "@/lib/prototype/scenarioLoader";
import { ConceptAView } from "./ConceptAView";
import { ConceptBView } from "./ConceptBView";
import { ConceptCView } from "./ConceptCView";

export type VariantType = "CONCEPT_A" | "CONCEPT_B" | "CONCEPT_C";

interface VariantRunnerProps {
  variant: VariantType;
  fixture: ScenarioFixture;
  onInteraction: (eventType: string, details: any) => void;
  onOrderAction: () => void;
  copyFeedback: string | null;
}

export function VariantRunner({ variant, fixture, onInteraction, onOrderAction, copyFeedback }: VariantRunnerProps) {
  switch (variant) {
    case "CONCEPT_A":
      return (
        <ConceptAView
          fixture={fixture}
          onInteraction={onInteraction}
          onOrderAction={onOrderAction}
          copyFeedback={copyFeedback}
        />
      );
    case "CONCEPT_B":
      return (
        <ConceptBView
          fixture={fixture}
          onInteraction={onInteraction}
          onOrderAction={onOrderAction}
          copyFeedback={copyFeedback}
        />
      );
    case "CONCEPT_C":
      return (
        <ConceptCView
          fixture={fixture}
          onInteraction={onInteraction}
          onOrderAction={onOrderAction}
          copyFeedback={copyFeedback}
        />
      );
    default:
      return (
        <div className="bg-rose-950/60 border border-rose-800 p-4 rounded text-rose-300 font-mono text-xs">
          [UNKNOWN VARIANT]: {variant}
        </div>
      );
  }
}
