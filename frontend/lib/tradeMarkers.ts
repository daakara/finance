/**
 * Canonical Trade Marker Classifier & Grouping Engine.
 *
 * Enforces Gate 1 & Gate 2 invariants:
 * - Maps SEC Form 4, Congressional STOCK Act, and Options Flow records strictly
 *   to their underlying source transactions:
 *     - P -> Purchase ("Open Market Purchase" ONLY when explicitly supported by source evidence)
 *     - S -> Sale ("Open Market Sale" ONLY when explicitly supported by source evidence)
 *     - A -> Grant / Award
 *     - F -> Tax Withholding
 *     - G -> Gift (Bona Fide Gift)
 *     - M / X / C / O -> Option Exercise / Derivative Conversion
 *     - UNKNOWN -> Unknown Form 4 Transaction (distinct from UNPARSED)
 *     - UNPARSED -> Form 4 Filing (Unparsed)
 * - Never infers buying conviction or execution venue from filing presence.
 * - Distinguishes retained SEC EDGAR evidence from curated/sample records.
 * - Prefers transaction_date over filing_date and marks dateSource explicitly.
 * - Never fabricates dates when source timestamps are missing.
 * - Groups same-bar / same-direction events while preserving access to every transaction & provenance.
 */

export type TradeMarkerCategory =
  | "PURCHASE"
  | "SALE"
  | "AWARD"
  | "WITHHOLDING"
  | "GIFT"
  | "EXERCISE"
  | "DISPOSITION"
  | "OTHER"
  | "MIXED"
  | "UNKNOWN"
  | "UNPARSED"
  | "CONGRESS_PURCHASE"
  | "CONGRESS_SALE"
  | "CONGRESS_OTHER"
  | "OPTIONS";

export type TradeMarkerDirection = "BUY" | "SELL" | "NEUTRAL";

export type TradeMarkerProvenance =
  | "RETAINED_SEC_EVIDENCE"
  | "VERIFIED_LIVE"
  | "CURATED_SAMPLE"
  | "UNVERIFIED";

export type TradeMarkerDateSource = "TRANSACTION_DATE" | "FILING_DATE";

export interface ChartTradeMarker {
  id: string;
  date: string;
  dateSource: TradeMarkerDateSource;
  transactionDate?: string | null;
  filingDate?: string | null;
  label: string;
  shortCode: string;
  type: "CONGRESS" | "INSIDER" | "OPTIONS";
  category: TradeMarkerCategory;
  direction: TradeMarkerDirection;
  transactionCode?: string | null;
  transactionType?: string | null;
  provenance: TradeMarkerProvenance;
  provenanceLabel: string;
  actor?: string | null;
  role?: string | null;
  shares?: number | null;
  amount?: string | number | null;
  accessionNumber?: string | null;
  secUrl?: string | null;
  isEffective?: boolean;
  effectiveStatus?: string | null;
}

export interface LegacyTradeMarkerInput {
  date: string;
  label: string;
  type?: "CONGRESS" | "INSIDER" | "OPTIONS";
  amount?: string | number | null;
}

export type AnyChartTradeMarker = ChartTradeMarker | LegacyTradeMarkerInput;

export interface GroupedCanvasMarker {
  time: string | number;
  position: "aboveBar" | "belowBar";
  color: string;
  shape: "arrowUp" | "arrowDown" | "circle" | "square";
  text: string;
  count: number;
  items: AnyChartTradeMarker[];
}

interface ClassificationResult {
  category: TradeMarkerCategory;
  direction: TradeMarkerDirection;
  shortCode: string;
  label: string;
}

const KNOWN_NEUTRAL_CODES: Record<string, ClassificationResult> = {
  A: { category: "AWARD", direction: "NEUTRAL", shortCode: "A", label: "Grant / Award" },
  F: { category: "WITHHOLDING", direction: "NEUTRAL", shortCode: "F", label: "Tax Withholding" },
  G: { category: "GIFT", direction: "NEUTRAL", shortCode: "G", label: "Gift" },
  M: { category: "EXERCISE", direction: "NEUTRAL", shortCode: "M", label: "Option Exercise" },
  X: { category: "EXERCISE", direction: "NEUTRAL", shortCode: "X", label: "Option Exercise" },
  C: { category: "EXERCISE", direction: "NEUTRAL", shortCode: "C", label: "Derivative Conversion" },
  O: { category: "EXERCISE", direction: "NEUTRAL", shortCode: "O", label: "Derivative Exercise" },
  D: { category: "DISPOSITION", direction: "NEUTRAL", shortCode: "D", label: "Disposition to Issuer" },
  J: { category: "OTHER", direction: "NEUTRAL", shortCode: "J", label: "Other Transaction (Code J)" },
  K: { category: "OTHER", direction: "NEUTRAL", shortCode: "K", label: "Equity Swap (Code K)" },
  U: { category: "OTHER", direction: "NEUTRAL", shortCode: "U", label: "Tender Offer (Code U)" },
  W: { category: "OTHER", direction: "NEUTRAL", shortCode: "W", label: "Inheritance / Will (Code W)" },
  V: { category: "OTHER", direction: "NEUTRAL", shortCode: "V", label: "Early Reporting (Code V)" },
  L: { category: "OTHER", direction: "NEUTRAL", shortCode: "L", label: "Small Acquisition (Code L)" },
  E: { category: "OTHER", direction: "NEUTRAL", shortCode: "E", label: "Short Expiration (Code E)" },
  H: { category: "OTHER", direction: "NEUTRAL", shortCode: "H", label: "Derivative Expiration (Code H)" },
  Z: { category: "OTHER", direction: "NEUTRAL", shortCode: "Z", label: "Voting Trust (Code Z)" },
};

function hasExplicitOpenMarketEvidence(record: Record<string, any>, upperType: string): boolean {
  return Boolean(
    record.is_open_market_purchase_evidence === true ||
    record.isOpenMarketPurchaseEvidence === true ||
    upperType.includes("OPEN MARKET") ||
    String(record.execution_venue ?? record.executionVenue ?? "").toUpperCase().includes("OPEN_MARKET")
  );
}

/**
 * Classifies a single SEC Form 4 record or transaction strictly from its source
 * code, explicit transaction type, and parsed metadata without assuming buys or venue.
 */
export function classifyForm4Transaction(record: Record<string, any>): ClassificationResult {
  const isMixed = Boolean(record.is_mixed ?? record.isMixed);
  const rawCode = String(record.transaction_code ?? record.transactionCode ?? "").trim().toUpperCase();
  const rawType = String(record.transaction_type ?? record.transactionType ?? "").trim();
  const upperType = rawType.toUpperCase();
  const effectiveStatus = String(record.effective_status ?? record.effectiveStatus ?? "").trim().toUpperCase();

  if (isMixed || rawCode === "MIXED") {
    return {
      category: "MIXED",
      direction: "NEUTRAL",
      shortCode: "MX",
      label: "Mixed Form 4 Filing",
    };
  }

  if (
    record.has_parsed_transaction === false ||
    effectiveStatus === "UNPARSED"
  ) {
    return {
      category: "UNPARSED",
      direction: "NEUTRAL",
      shortCode: "F4",
      label: "Form 4 Filing (Unparsed)",
    };
  }

  if (rawCode === "UNKNOWN" || upperType === "UNKNOWN") {
    return {
      category: "UNKNOWN",
      direction: "NEUTRAL",
      shortCode: "?",
      label: "Unknown Form 4 Transaction",
    };
  }

  if (rawCode === "P") {
    // Guard against curated descriptions like "Purchase (Grant / Long Hold)"
    if (upperType.includes("GRANT") || upperType.includes("AWARD")) {
      return KNOWN_NEUTRAL_CODES.A;
    }
    const openMarket = hasExplicitOpenMarketEvidence(record, upperType);
    return {
      category: "PURCHASE",
      direction: "BUY",
      shortCode: "P",
      label: openMarket ? "Open Market Purchase" : "Purchase",
    };
  }

  if (rawCode === "S") {
    const openMarket = upperType.includes("OPEN MARKET");
    return {
      category: "SALE",
      direction: "SELL",
      shortCode: "S",
      label: openMarket ? "Open Market Sale" : "Sale",
    };
  }

  if (rawCode && KNOWN_NEUTRAL_CODES[rawCode]) {
    return KNOWN_NEUTRAL_CODES[rawCode];
  }

  if (rawCode) {
    return {
      category: "UNKNOWN",
      direction: "NEUTRAL",
      shortCode: rawCode,
      label: `Unknown Form 4 Transaction (Code ${rawCode})`,
    };
  }

  // Fallback on text/flags only when transaction_code is absent
  if (upperType.includes("TAX") || upperType.includes("WITHHOLD")) {
    return KNOWN_NEUTRAL_CODES.F;
  }
  if (record.is_gift === true || record.isGift === true || upperType.includes("GIFT")) {
    return KNOWN_NEUTRAL_CODES.G;
  }
  if (upperType.includes("GRANT") || upperType.includes("AWARD") || upperType.includes("RSU")) {
    return KNOWN_NEUTRAL_CODES.A;
  }
  if (
    record.is_exercise === true ||
    record.isExercise === true ||
    upperType.includes("EXERCISE") ||
    upperType.includes("CONVERSION")
  ) {
    return KNOWN_NEUTRAL_CODES.M;
  }
  if (upperType.includes("DISPOSITION TO ISSUER")) {
    return KNOWN_NEUTRAL_CODES.D;
  }
  if (record.is_sell === true || record.isSell === true || upperType.includes("SALE") || upperType.includes("SELL")) {
    return {
      category: "SALE",
      direction: "SELL",
      shortCode: "S",
      label: upperType.includes("OPEN MARKET") ? "Open Market Sale" : "Sale",
    };
  }
  if (record.is_buy === true || record.isBuy === true || upperType.includes("PURCHASE") || upperType.includes("BUY")) {
    const openMarket = hasExplicitOpenMarketEvidence(record, upperType);
    return {
      category: "PURCHASE",
      direction: "BUY",
      shortCode: "P",
      label: openMarket ? "Open Market Purchase" : "Purchase",
    };
  }
  if (rawType) {
    return {
      category: "UNKNOWN",
      direction: "NEUTRAL",
      shortCode: "?",
      label: rawType,
    };
  }

  if (record.has_parsed_transaction === true || record.transaction_id || record.transactionId) {
    return {
      category: "UNKNOWN",
      direction: "NEUTRAL",
      shortCode: "?",
      label: "Unknown Form 4 Transaction",
    };
  }

  return {
    category: "UNPARSED",
    direction: "NEUTRAL",
    shortCode: "F4",
    label: "Form 4 Filing (Unparsed)",
  };
}

function resolveSecProvenance(
  filing: Record<string, any>,
  secSubdomain?: Record<string, any> | null
): { provenance: TradeMarkerProvenance; provenanceLabel: string } {
  const subStatus = String(secSubdomain?.status ?? "").toUpperCase();
  const subProv = String(secSubdomain?.provenance ?? "").toLowerCase();
  const accNum = filing.accession_number ?? filing.accessionNumber;
  const secUrl = String(filing.sec_url ?? filing.secUrl ?? "");
  const idStr = String(filing.id ?? "");

  const explicitlyCurated = Boolean(
    filing.is_curated === true ||
    filing.isCurated === true ||
    String(filing.source_type ?? "").toUpperCase() === "CURATED" ||
    subStatus === "CURATED_RESEARCH" ||
    subProv.includes("curated") ||
    (!accNum && idStr.startsWith("sec-") && !secUrl.includes("/Archives/edgar/"))
  );

  if (explicitlyCurated) {
    return { provenance: "CURATED_SAMPLE", provenanceLabel: "Curated Sample" };
  }

  const hasRetainedEvidence = Boolean(
    accNum ||
    filing.xml_hash ||
    secUrl.includes("/Archives/edgar/") ||
    subStatus === "PIPELINE_HELD" ||
    subProv.includes("sec edgar")
  );

  if (hasRetainedEvidence) {
    if (subStatus === "VERIFIED_SIGNAL" && secSubdomain?.is_live === true) {
      return { provenance: "VERIFIED_LIVE", provenanceLabel: "Live SEC Form 4" };
    }
    return { provenance: "RETAINED_SEC_EVIDENCE", provenanceLabel: "Retained SEC Form 4" };
  }

  return { provenance: "UNVERIFIED", provenanceLabel: "Unverified Source" };
}

function resolveMarkerDate(
  txDateRaw: unknown,
  filingDateRaw: unknown
): { date: string; dateSource: TradeMarkerDateSource; transactionDate: string | null; filingDate: string | null } | null {
  const txDate = typeof txDateRaw === "string" && txDateRaw.trim() ? txDateRaw.trim() : null;
  const filingDate = typeof filingDateRaw === "string" && filingDateRaw.trim() ? filingDateRaw.trim() : null;

  if (txDate) {
    return { date: txDate, dateSource: "TRANSACTION_DATE", transactionDate: txDate, filingDate };
  }
  if (filingDate) {
    return { date: filingDate, dateSource: "FILING_DATE", transactionDate: null, filingDate };
  }
  return null;
}

function formatSharesOrValue(shares: unknown, totalValue: unknown): string | number | null {
  if (typeof totalValue === "number" && Number.isFinite(totalValue) && totalValue > 0) {
    return totalValue >= 1_000_000
      ? `$${(totalValue / 1_000_000).toFixed(2)}M`
      : totalValue >= 1_000
      ? `$${Math.round(totalValue / 1_000)}K`
      : `$${totalValue.toFixed(0)}`;
  }
  if (typeof totalValue === "string" && totalValue.trim()) {
    return totalValue.trim();
  }
  if (typeof shares === "number" && Number.isFinite(shares) && shares > 0) {
    return `${shares.toLocaleString()} sh`;
  }
  return null;
}

function resolveEffectiveClassification(
  base: ClassificationResult,
  record: Record<string, any>
): { classification: ClassificationResult; isEffective: boolean; effectiveStatus: string | null } {
  const isEff = typeof record.is_effective === "boolean"
    ? record.is_effective
    : typeof record.isEffective === "boolean"
    ? record.isEffective
    : true;
  const effStatus = String(record.effective_status ?? record.effectiveStatus ?? "").trim() || null;

  if (!isEff && effStatus && effStatus.toUpperCase() !== "EFFECTIVE" && effStatus.toUpperCase() !== "EFFECTIVE_REPLACEMENT") {
    const statusSuffix =
      effStatus.toUpperCase() === "SUPERSEDED_BY_AMENDMENT"
        ? "Superseded"
        : effStatus.toUpperCase() === "PENDING_PUBLIC_AVAILABILITY"
        ? "Pending Cutoff"
        : effStatus.toUpperCase() === "SCOPE_UNVERIFIED" || effStatus.toUpperCase() === "AMENDMENT_EFFECT_AMBIGUOUS"
        ? "Scope Unverified"
        : effStatus;
    return {
      classification: {
        ...base,
        direction: "NEUTRAL",
        label: `${base.label} (${statusSuffix})`,
      },
      isEffective: false,
      effectiveStatus: effStatus,
    };
  }

  return { classification: base, isEffective: isEff, effectiveStatus: effStatus };
}

/**
 * Builds normalized, provenance-aware trade markers from the analytics smartMoney payload.
 */
export function buildChartTradeMarkers(smartMoney: Record<string, any> | null | undefined): ChartTradeMarker[] {
  if (!smartMoney || typeof smartMoney !== "object") return [];

  const markers: ChartTradeMarker[] = [];
  const subdomains = smartMoney.subdomains ?? {};
  const secSub = subdomains.sec_form_4 ?? smartMoney.feeds?.sec_form4 ?? null;
  const congSub = subdomains.congressional_stock_act ?? smartMoney.feeds?.congressional_disclosures ?? null;
  const optSub = subdomains.options_flow ?? smartMoney.feeds?.options_flow ?? null;

  // 1. Congressional STOCK Act Disclosures
  const congList = Array.isArray(smartMoney.congressTrades) ? smartMoney.congressTrades : [];
  const congCurated =
    String(congSub?.status ?? "").toUpperCase() === "CURATED_RESEARCH" ||
    String(congSub?.provenance ?? "").toLowerCase().includes("curated");

  congList.forEach((ct: any, idx: number) => {
    const resolvedDate = resolveMarkerDate(ct?.transaction_date || ct?.date, ct?.filing_date);
    if (!resolvedDate) return;

    const rawTxType = String(ct.transaction_type || ct.type || "").trim();
    const upperTx = rawTxType.toUpperCase();
    const isSale = upperTx.includes("SALE") || upperTx.includes("SELL");
    const isPurchase = !isSale && (upperTx.includes("PURCHASE") || upperTx.includes("BUY"));

    const category: TradeMarkerCategory = isSale
      ? "CONGRESS_SALE"
      : isPurchase
      ? "CONGRESS_PURCHASE"
      : "CONGRESS_OTHER";
    const direction: TradeMarkerDirection = isSale ? "SELL" : isPurchase ? "BUY" : "NEUTRAL";
    const shortCode = isSale ? "CS" : isPurchase ? "CP" : "CG";
    const actionWord = isSale ? "Sale" : isPurchase ? "Purchase" : rawTxType || "Disclosure";
    const lastName = ct.politician?.split(" ")?.slice(-1)[0] || "Congress";

    const isItemCurated = congCurated || String(ct.id ?? "").startsWith("cong-");
    markers.push({
      id: String(ct.id || `cong-${idx}-${resolvedDate.date}`),
      date: resolvedDate.date,
      dateSource: resolvedDate.dateSource,
      transactionDate: resolvedDate.transactionDate,
      filingDate: resolvedDate.filingDate,
      label: `Congress ${actionWord}: ${lastName}`,
      shortCode,
      type: "CONGRESS",
      category,
      direction,
      transactionType: rawTxType || null,
      provenance: isItemCurated ? "CURATED_SAMPLE" : "VERIFIED_LIVE",
      provenanceLabel: isItemCurated ? "Curated Sample" : "Official STOCK Act",
      actor: ct.politician || null,
      role: ct.chamber ? `${ct.chamber} Disclosure` : "Congressional Disclosure",
      amount: ct.amount_range || null,
      isEffective: true,
    });
  });

  // 2. SEC Form 4 Insider Filings / Parsed Transactions
  const secList: any[] =
    (Array.isArray(smartMoney.secForm4Trades) && smartMoney.secForm4Trades.length > 0
      ? smartMoney.secForm4Trades
      : Array.isArray(smartMoney.secInsiderTrades)
      ? smartMoney.secInsiderTrades
      : []) as any[];

  secList.forEach((filing: any, fIdx: number) => {
    if (!filing || typeof filing !== "object") return;
    const { provenance, provenanceLabel } = resolveSecProvenance(filing, secSub);
    const accNum = filing.accession_number || filing.accessionNumber || null;
    const secUrl = filing.sec_url || filing.secUrl || null;
    const filingDateFallback = filing.filing_date || filing.filingDate || filing.date || null;
    const parsedTxs = Array.isArray(filing.transactions) ? filing.transactions : [];

    if (parsedTxs.length > 0) {
      parsedTxs.forEach((tx: any, tIdx: number) => {
        const resolvedDate = resolveMarkerDate(
          tx.transaction_date || tx.transactionDate || filing.transaction_date || filing.transactionDate,
          filingDateFallback
        );
        if (!resolvedDate) return;

        const baseCls = classifyForm4Transaction({ ...tx, has_parsed_transaction: true });
        const { classification, isEffective, effectiveStatus } = resolveEffectiveClassification(baseCls, tx);

        markers.push({
          id: String(tx.transaction_id || tx.transactionId || `${accNum || fIdx}-${tIdx}`),
          date: resolvedDate.date,
          dateSource: resolvedDate.dateSource,
          transactionDate: resolvedDate.transactionDate,
          filingDate: resolvedDate.filingDate,
          label: classification.label,
          shortCode: classification.shortCode,
          type: "INSIDER",
          category: classification.category,
          direction: classification.direction,
          transactionCode: tx.transaction_code || tx.transactionCode || null,
          transactionType: tx.transaction_type || tx.transactionType || null,
          provenance,
          provenanceLabel,
          actor: tx.reporting_owner_name || tx.reportingOwnerName || filing.insider_name || filing.insiderName || null,
          role: tx.officer_title || tx.officerTitle || tx.reporting_owner_role || tx.reportingOwnerRole || filing.insider_role || filing.role || null,
          shares: typeof tx.shares === "number" && Number.isFinite(tx.shares) ? tx.shares : null,
          amount: formatSharesOrValue(tx.shares, tx.total_value_usd ?? tx.totalValueUsd),
          accessionNumber: accNum,
          secUrl,
          isEffective,
          effectiveStatus,
        });
      });
      return;
    }

    const resolvedDate = resolveMarkerDate(
      filing.transaction_date || filing.transactionDate,
      filingDateFallback
    );
    if (!resolvedDate) return;

    const baseCls = classifyForm4Transaction(filing);
    const { classification, isEffective, effectiveStatus } = resolveEffectiveClassification(baseCls, filing);

    markers.push({
      id: String(filing.id || accNum || `sec-${fIdx}-${resolvedDate.date}`),
      date: resolvedDate.date,
      dateSource: resolvedDate.dateSource,
      transactionDate: resolvedDate.transactionDate,
      filingDate: resolvedDate.filingDate,
      label: classification.label,
      shortCode: classification.shortCode,
      type: "INSIDER",
      category: classification.category,
      direction: classification.direction,
      transactionCode: filing.transaction_code || filing.transactionCode || null,
      transactionType: filing.transaction_type || filing.transactionType || null,
      provenance,
      provenanceLabel,
      actor: filing.insider_name || filing.insiderName || null,
      role: filing.insider_role || filing.insiderRole || filing.role || null,
      shares: typeof filing.shares_traded === "number" && Number.isFinite(filing.shares_traded) ? filing.shares_traded : null,
      amount: formatSharesOrValue(filing.shares_traded ?? filing.shares, filing.total_value_usd ?? filing.total_value ?? filing.totalValueUsd ?? filing.transactionValue),
      accessionNumber: accNum,
      secUrl,
      isEffective,
      effectiveStatus,
    });
  });

  // 3. Unusual Options Flow (never fabricate missing dates)
  const optList = Array.isArray(smartMoney.optionsFlow) ? smartMoney.optionsFlow : [];
  const optCurated =
    String(optSub?.status ?? "").toUpperCase() === "CURATED_RESEARCH" ||
    String(optSub?.provenance ?? "").toLowerCase().includes("curated");

  optList.forEach((op: any, idx: number) => {
    const rawDate = op?.date || op?.timestamp;
    if (!rawDate) return;

    const flowType = String(op.type || op.put_call || "Options Flow").trim();
    const sentiment = String(op.sentiment || "").toUpperCase();
    const isPut = flowType.toUpperCase().includes("PUT") || sentiment.includes("BEAR");
    const isCall = !isPut && (flowType.toUpperCase().includes("CALL") || sentiment.includes("BULL"));

    markers.push({
      id: String(op.id || `opt-${idx}-${rawDate}`),
      date: String(rawDate),
      dateSource: "TRANSACTION_DATE",
      transactionDate: String(rawDate),
      filingDate: null,
      label: flowType,
      shortCode: isPut ? "PUT" : isCall ? "CALL" : "OPT",
      type: "OPTIONS",
      category: "OPTIONS",
      direction: isPut ? "SELL" : isCall ? "BUY" : "NEUTRAL",
      transactionType: flowType,
      provenance: optCurated ? "CURATED_SAMPLE" : "VERIFIED_LIVE",
      provenanceLabel: optCurated ? "Curated Sample" : "Options Flow",
      amount: op.premium || null,
      isEffective: true,
    });
  });

  return markers;
}

function isFullChartMarker(m: AnyChartTradeMarker): m is ChartTradeMarker {
  return "category" in m && "direction" in m;
}

function inferDirectionFromLegacy(m: LegacyTradeMarkerInput): TradeMarkerDirection {
  const upper = m.label.toUpperCase();
  if (upper.includes("SALE") || upper.includes("SELL") || upper.includes("PUT")) return "SELL";
  if (upper.includes("GIFT") || upper.includes("AWARD") || upper.includes("GRANT") || upper.includes("TAX") || upper.includes("WITHHOLD") || upper.includes("EXERCISE") || upper.includes("MIXED") || upper.includes("UNKNOWN") || upper.includes("UNPARSED")) {
    return "NEUTRAL";
  }
  if (upper.includes("PURCHASE") || upper.includes("BUY") || upper.includes("CALL")) return "BUY";
  return "NEUTRAL";
}

/**
 * Aligns markers to valid candle timestamps and groups same-candle / same-direction
 * events into clean, non-overlapping Lightweight Charts markers while preserving full item lists.
 */
export function groupTradeMarkersForChart(
  tradeMarkers: AnyChartTradeMarker[],
  candleTimes: Array<string | number>,
  compact: boolean
): GroupedCanvasMarker[] {
  if (!tradeMarkers.length || !candleTimes.length) return [];

  const groups = new Map<
    string,
    {
      time: string | number;
      direction: TradeMarkerDirection;
      hasOptions: boolean;
      items: AnyChartTradeMarker[];
    }
  >();

  for (const tm of tradeMarkers) {
    if (!tm.date) continue;
    const targetDateStr = tm.date.split("T")[0];

    let matchedTime = candleTimes.find((ct) => typeof ct === "string" && ct === targetDateStr);
    if (!matchedTime) {
      const targetMs = new Date(targetDateStr).getTime();
      if (!Number.isFinite(targetMs)) continue;
      let closest = candleTimes[0];
      let minDiff = Infinity;
      for (const ct of candleTimes) {
        const ctMs = typeof ct === "number" ? (ct > 20000000000 ? ct : ct * 1000) : new Date(ct).getTime();
        const diff = Math.abs(ctMs - targetMs);
        if (diff < minDiff) {
          minDiff = diff;
          closest = ct;
        }
      }
      if (minDiff <= 14 * 24 * 60 * 60 * 1000) {
        matchedTime = closest;
      }
    }

    if (!matchedTime) continue;

    const direction = isFullChartMarker(tm) ? tm.direction : inferDirectionFromLegacy(tm);
    const isOptions = tm.type === "OPTIONS";
    const groupKey = `${String(matchedTime)}::${isOptions ? "OPT" : direction}`;

    const existing = groups.get(groupKey);
    if (existing) {
      existing.items.push(tm);
    } else {
      groups.set(groupKey, {
        time: matchedTime,
        direction,
        hasOptions: isOptions,
        items: [tm],
      });
    }
  }

  const canvasMarkers: GroupedCanvasMarker[] = [];

  for (const grp of groups.values()) {
    const count = grp.items.length;
    const first = grp.items[0];
    const isBuy = grp.direction === "BUY";
    const isSell = grp.direction === "SELL";

    const position: "aboveBar" | "belowBar" = isBuy ? "belowBar" : "aboveBar";
    const color = grp.hasOptions
      ? "#f59e0b"
      : isBuy
      ? "#10b981"
      : isSell
      ? "#f43f5e"
      : "#94a3b8";
    const shape: "arrowUp" | "arrowDown" | "circle" | "square" = grp.hasOptions
      ? "circle"
      : isBuy
      ? "arrowUp"
      : isSell
      ? "arrowDown"
      : "circle";

    let text = "";
    if (!compact) {
      if (isFullChartMarker(first)) {
        const uniqueCodes = Array.from(new Set(grp.items.map((item) => (isFullChartMarker(item) ? item.shortCode : "")).filter(Boolean)));
        if (count === 1) {
          text = `${first.shortCode}: ${first.label}`;
        } else if (uniqueCodes.length === 1) {
          text = `${first.shortCode} ×${count}`;
        } else {
          text = `F4 ×${count}`;
        }
      } else {
        text = count === 1 ? first.label : `${first.label} ×${count}`;
      }
    }

    canvasMarkers.push({
      time: grp.time,
      position,
      color,
      shape,
      text,
      count,
      items: grp.items,
    });
  }

  canvasMarkers.sort((a, b) => {
    const tA = typeof a.time === "number" ? a.time : new Date(String(a.time) + "T00:00:00Z").getTime();
    const tB = typeof b.time === "number" ? b.time : new Date(String(b.time) + "T00:00:00Z").getTime();
    return tA - tB;
  });

  return canvasMarkers;
}
