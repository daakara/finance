"use client";

import React, { useState } from "react";
import { useExperienceMode } from "../context/ExperienceModeContext";
import { generateQuantitativeInsight } from "../lib/insightGenerator";
import GuidedTerminalView from "./terminal/GuidedTerminalView";
import StandardTerminalView from "./terminal/StandardTerminalView";
import AdvancedTerminalView from "./terminal/AdvancedTerminalView";
import WhyInspectModal from "./WhyInspectModal";
import PositionSizerModal from "./PositionSizerModal";

interface AdaptiveTerminalProps {
  symbol: string;
  companyName?: string;
  currentPrice: number;
  changePct: number;
  setupScore?: number;
  isStage4?: boolean;
}

export default function AdaptiveTerminal({
  symbol,
  companyName = "Asset Intelligence",
  currentPrice,
  changePct,
  setupScore = 60,
  isStage4 = false,
}: AdaptiveTerminalProps) {
  const { experienceMode } = useExperienceMode();
  const [isWhyOpen, setIsWhyOpen] = useState(false);
  const [isSizerOpen, setIsSizerOpen] = useState(false);

  const insight = generateQuantitativeInsight(
    symbol,
    companyName,
    currentPrice,
    changePct,
    setupScore,
    isStage4 ? 4 : 2
  );

  return (
    <div className="w-full space-y-4 font-sans">
      {experienceMode === "GUIDED" && (
        <GuidedTerminalView
          insight={insight}
          onOpenSizer={() => setIsSizerOpen(true)}
          onOpenWhy={() => setIsWhyOpen(true)}
        />
      )}

      {experienceMode === "STANDARD" && (
        <StandardTerminalView
          insight={insight}
          onOpenSizer={() => setIsSizerOpen(true)}
          onOpenWhy={() => setIsWhyOpen(true)}
        />
      )}

      {experienceMode === "ADVANCED" && (
        <AdvancedTerminalView
          insight={insight}
          onOpenSizer={() => setIsSizerOpen(true)}
          onOpenWhy={() => setIsWhyOpen(true)}
        />
      )}

      {/* Why Score Attribution Modal */}
      <WhyInspectModal
        isOpen={isWhyOpen}
        onClose={() => setIsWhyOpen(false)}
        symbol={symbol}
        setupScore={insight.setupScore}
        items={insight.scoreAttribution.items}
        catalystToIncreaseScore={insight.scoreAttribution.catalystToIncreaseScore}
      />

      {/* Institutional Position Sizer Modal */}
      <PositionSizerModal
        isOpen={isSizerOpen}
        onClose={() => setIsSizerOpen(false)}
        symbol={symbol}
        entryPrice={currentPrice}
        stopLoss={insight.standard.keyLevels.stopLoss}
        takeProfit1={insight.standard.keyLevels.target1}
        isStage4={isStage4}
      />
    </div>
  );
}
