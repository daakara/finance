/**
 * Order Clipboard Utility
 *
 * Formats order plan text deterministically and writes to clipboard without side-effects.
 * Ensures zero journal, portfolio, execution, or network persistence side-effects.
 */

export interface FormatOrderPlanParams {
  recommendedShares: number;
  ticker: string;
  entryPivot: number;
  stopLoss: number;
  target1?: number | null;
}

export function formatOrderPlanString(params: FormatOrderPlanParams): string {
  const t1 = params.target1 !== null && params.target1 !== undefined && params.target1 > 0
    ? `$${params.target1.toFixed(2)}`
    : "--";
  return `BUY ${params.recommendedShares} ${params.ticker} LMT $${params.entryPivot.toFixed(2)} | STP $${params.stopLoss.toFixed(2)} | TGT ${t1}`;
}

export interface ClipboardWriter {
  writeText: (text: string) => Promise<void>;
}

export async function copyOrderPlanToClipboard(
  orderStr: string,
  clipboardApi?: ClipboardWriter
): Promise<{ success: boolean; error?: string }> {
  try {
    if (!clipboardApi?.writeText) {
      throw new Error("Clipboard API unavailable in this browser context.");
    }
    await clipboardApi.writeText(orderStr);
    return { success: true };
  } catch (err: any) {
    return {
      success: false,
      error: err?.message || "Clipboard write permission denied.",
    };
  }
}
