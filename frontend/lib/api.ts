export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export interface AnalyticsResponse {
  symbol: string;
  period: string;
  analytics: {
    advanced_metrics?: {
      VaR_95?: number;
      VaR_99?: number;
      Modified_VaR_95?: number;
      Modified_VaR_99?: number;
      Modified_CVaR_95?: number;
      Sortino_Ratio?: number;
      Calmar_Ratio?: number;
      Skewness?: number;
      Kurtosis?: number;
    };
    drawdown_analysis?: {
      max_drawdown?: number;
    };
    regime_analysis?: {
      current_regime?: string;
    };
  };
}

export interface VolatilityForecastResponse {
  symbol: string;
  horizon_days: number;
  forecast: {
    price_forecast?: {
      last_price: number;
      predicted_prices: number[];
      expected_change_pct: number;
      model_type: string;
    };
    out_of_sample_evaluation?: {
      rmse: number;
      qlike_loss: number;
      mae: number;
      status: string;
    };
  };
}

export interface GemCandidate {
  ticker: string;
  composite_score: number;
  risk_rating: string;
  investment_thesis: string;
  primary_catalyst: string;
}

export interface ScreenerResponse {
  total_candidates: number;
  gems_found: number;
  results: GemCandidate[];
}

export async function fetchAssetAnalytics(symbol: string): Promise<AnalyticsResponse> {
  const res = await fetch(`${API_BASE_URL}/analytics/${symbol}`);
  if (!res.ok) throw new Error(`Failed to fetch analytics for ${symbol}`);
  return res.json();
}

export async function fetchVolatilityForecast(symbol: string, horizon: number = 30): Promise<VolatilityForecastResponse> {
  const res = await fetch(`${API_BASE_URL}/volatility/${symbol}?horizon=${horizon}`);
  if (!res.ok) throw new Error(`Failed to fetch volatility forecast for ${symbol}`);
  return res.json();
}

export async function runHiddenGemsScreener(tickers: string[]): Promise<ScreenerResponse> {
  const res = await fetch(`${API_BASE_URL}/screener/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tickers }),
  });
  if (!res.ok) throw new Error("Failed to run hidden gems screener");
  return res.json();
}

