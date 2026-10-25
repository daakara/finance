// ARX Ephemeral Synthesis E Fixtures Authority (Candidate V2 Enriched)
const ARX_SYNTHESIS_FIXTURES = {
  "account_context": {
    "accountEquity": 124500,
    "standardRiskBudgetPct": 0.02,
    "maxRiskDollarBudget": 2490
  },
  "states": {
    "NVDA": {
      "state_id": "STATE_1_ACTIONABLE",
      "symbol": "NVDA",
      "name": "NVIDIA Corporation",
      "exchange": "NASDAQ",
      "sector": "Large Cap Semiconductors",
      "volumeStatement": "54.2M (148% vs 50D avg)",
      "currentPrice": 126.4,
      "priceDeltaDollar": 4.12,
      "priceDeltaPct": 3.37,
      "decisionState": "ACTIONABLE_SETUP",
      "executionStatus": "IN_BUY_ZONE",
      "isActionable": true,
      "pattern": "Stage 2 VCP Breakout",
      "confluenceScore": 88,
      "confluenceRating": "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT",
      "confluenceBadge": "4-Pillar Confluence (Pristine)",
      "levels": {
        "pivot": 125.8,
        "entryMin": 124.5,
        "entryMax": 126.8,
        "referencePrice": 126.4,
        "stopLoss": 121.5,
        "stopDistanceDollar": 4.9,
        "stopDistancePct": -3.88,
        "target1": 136.2,
        "rewardDistanceDollar": 9.8,
        "rewardDistancePct": 7.75,
        "rewardRiskRatio": 2,
        "riskRewardDisplay": "1 : 2.00",
        "rMultipleTarget1": 2
      },
      "sizing": {
        "shares": 100,
        "quantityAuthority": "ILLUSTRATIVE_TEST_LOT",
        "generatedByCanonicalSizingEngine": false,
        "capitalAllocated": 12640,
        "portfolioAllocationPct": 10.15,
        "modeledStopLossDollar": 490,
        "modeledStopLossEquityPct": 0.394,
        "riskBudgetDollar": 2490,
        "riskBudgetUtilizationPct": 19.68,
        "bindingConstraint": "Illustrative Simulation Lot (100 Shares - Unconstrained 2% risk permits 508 shares)",
        "disclaimer": "Modeled loss to stop: -$490.00 (-3.88%). Stop orders do not guarantee execution price against market gaps or overnight slippage."
      },
      "pillars": [
        {
          "pillar": "TECHNICAL_STRUCTURE",
          "label": "Technical Confluence",
          "weightPct": 25,
          "score": 92,
          "status": "positive",
          "detail": "Pivot breakout at $125.80; 3 tightening volatility contractions (14% \u2192 7% \u2192 2.8%); 20 EMA trending up."
        },
        {
          "pillar": "FUNDAMENTAL_SOLVENCY",
          "label": "Fundamental Moat",
          "weightPct": 25,
          "score": 89,
          "status": "positive",
          "detail": "Piotroski F-Score 9/9; ROIC 42.5%; gross margins expanding +180 bps."
        },
        {
          "pillar": "SMART_MONEY_FLOW",
          "label": "Institutional Flow",
          "weightPct": 25,
          "score": 86,
          "status": "positive",
          "detail": "Volume +48% above 50-day average; Up/down volume ratio 1.84; pocket pivot accumulation signal confirmed."
        },
        {
          "pillar": "MACRO_SAFETY_FLOOR",
          "label": "Macro Environment",
          "weightPct": 25,
          "score": 85,
          "status": "positive",
          "detail": "S&P 500 confirmed uptrend above 21 EMA; 10Y yields declining (-3 bps); VIX muted at 15.20."
        }
      ],
      "provenance": {
        "asOf": "2026-10-06 (Static Baseline Fixture)",
        "authority": "ARX Ephemeral Prototype Fixture Authority (Offline Simulation)",
        "quoteFreshness": "STATIC_OFFLINE_PROTOTYPE",
        "dataMode": "ILLUSTRATIVE_SYNTHETIC_FIXTURE",
        "marketDataFreshness": "NOT_APPLICABLE_TO_STATIC_PROTOTYPE"
      },
      "secType": "COMMON_STOCK",
      "security_type": "COMMON_STOCK",
      "display_mode": "SINGLE_STOCK_ANALYSIS",
      "executionEligibility": "STOCK_EXECUTION",
      "candles": [
        {
          "time": "2026-08-11",
          "open": 115.43,
          "high": 117.32,
          "low": 114.47,
          "close": 116.36,
          "volume": 35000000
        },
        {
          "time": "2026-08-12",
          "open": 115.49,
          "high": 118.29,
          "low": 114.19,
          "close": 116.99,
          "volume": 49874972
        },
        {
          "time": "2026-08-13",
          "open": 116.13,
          "high": 119.65,
          "low": 114.56,
          "close": 118.08,
          "volume": 31166883
        },
        {
          "time": "2026-08-14",
          "open": 117.22,
          "high": 121.09,
          "low": 115.52,
          "close": 119.39,
          "volume": 21112780
        },
        {
          "time": "2026-08-17",
          "open": 118.53,
          "high": 122.29,
          "low": 116.87,
          "close": 120.63,
          "volume": 42411700
        },
        {
          "time": "2026-08-18",
          "open": 119.76,
          "high": 122.96,
          "low": 118.31,
          "close": 121.51,
          "volume": 46977307
        },
        {
          "time": "2026-08-19",
          "open": 120.65,
          "high": 122.96,
          "low": 119.53,
          "close": 121.84,
          "volume": 24501880
        },
        {
          "time": "2026-08-20",
          "open": 120.98,
          "high": 122.29,
          "low": 120.24,
          "close": 121.55,
          "volume": 25727943
        },
        {
          "time": "2026-08-21",
          "open": 120.68,
          "high": 121.17,
          "low": 120.25,
          "close": 120.74,
          "volume": 47887427
        },
        {
          "time": "2026-08-24",
          "open": 119.88,
          "high": 120.42,
          "low": 119.11,
          "close": 119.65,
          "volume": 40951109
        },
        {
          "time": "2026-08-25",
          "open": 118.79,
          "high": 119.31,
          "low": 118.07,
          "close": 118.59,
          "volume": 20579038
        },
        {
          "time": "2026-08-26",
          "open": 117.72,
          "high": 118.31,
          "low": 117.25,
          "close": 117.84,
          "volume": 32765015
        },
        {
          "time": "2026-08-27",
          "open": 116.98,
          "high": 118.45,
          "low": 116.18,
          "close": 117.65,
          "volume": 49996894
        },
        {
          "time": "2026-08-28",
          "open": 116.79,
          "high": 119.28,
          "low": 115.61,
          "close": 118.1,
          "volume": 33370451
        },
        {
          "time": "2026-08-31",
          "open": 117.24,
          "high": 120.68,
          "low": 115.7,
          "close": 119.14,
          "volume": 20423023
        },
        {
          "time": "2026-09-01",
          "open": 118.27,
          "high": 122.35,
          "low": 116.49,
          "close": 120.57,
          "volume": 40385875
        },
        {
          "time": "2026-09-02",
          "open": 119.71,
          "high": 123.95,
          "low": 117.87,
          "close": 122.11,
          "volume": 48189096
        },
        {
          "time": "2026-09-03",
          "open": 121.25,
          "high": 125.15,
          "low": 119.54,
          "close": 123.44,
          "volume": 26215440
        },
        {
          "time": "2026-09-04",
          "open": 122.57,
          "high": 125.66,
          "low": 121.16,
          "close": 124.25,
          "volume": 24074588
        },
        {
          "time": "2026-09-07",
          "open": 123.39,
          "high": 125.42,
          "low": 122.38,
          "close": 124.41,
          "volume": 46599918
        },
        {
          "time": "2026-09-08",
          "open": 123.55,
          "high": 124.53,
          "low": 122.93,
          "close": 123.91,
          "volume": 42936240
        },
        {
          "time": "2026-09-09",
          "open": 123.04,
          "high": 123.51,
          "low": 122.46,
          "close": 122.93,
          "volume": 21355000
        },
        {
          "time": "2026-09-10",
          "open": 122.07,
          "high": 122.64,
          "low": 121.22,
          "close": 121.79,
          "volume": 30579926
        },
        {
          "time": "2026-09-11",
          "open": 120.93,
          "high": 121.39,
          "low": 120.36,
          "close": 120.82,
          "volume": 49784004
        },
        {
          "time": "2026-09-14",
          "open": 119.96,
          "high": 120.95,
          "low": 119.34,
          "close": 120.33,
          "volume": 35610399
        },
        {
          "time": "2026-09-15",
          "open": 119.46,
          "high": 121.52,
          "low": 118.44,
          "close": 120.5,
          "volume": 20058702
        },
        {
          "time": "2026-09-16",
          "open": 119.64,
          "high": 122.83,
          "low": 118.19,
          "close": 121.38,
          "volume": 38239809
        },
        {
          "time": "2026-09-17",
          "open": 120.52,
          "high": 124.63,
          "low": 118.73,
          "close": 122.84,
          "volume": 49106434
        },
        {
          "time": "2026-09-18",
          "open": 121.97,
          "high": 126.58,
          "low": 119.99,
          "close": 124.6,
          "volume": 28125118
        },
        {
          "time": "2026-09-21",
          "open": 123.74,
          "high": 128.3,
          "low": 121.78,
          "close": 126.34,
          "volume": 22665147
        },
        {
          "time": "2026-09-22",
          "open": 125.48,
          "high": 129.41,
          "low": 123.76,
          "close": 127.69,
          "volume": 45053438
        },
        {
          "time": "2026-09-23",
          "open": 126.82,
          "high": 129.7,
          "low": 125.49,
          "close": 128.37,
          "volume": 44744193
        },
        {
          "time": "2026-09-24",
          "open": 127.51,
          "high": 129.19,
          "low": 126.63,
          "close": 128.31,
          "volume": 22435591
        },
        {
          "time": "2026-09-25",
          "open": 127.45,
          "high": 128.07,
          "low": 126.97,
          "close": 127.59,
          "volume": 28493517
        },
        {
          "time": "2026-09-28",
          "open": 126.73,
          "high": 127.28,
          "low": 125.93,
          "close": 126.48,
          "volume": 49241058
        },
        {
          "time": "2026-09-29",
          "open": 125.61,
          "high": 126.18,
          "low": 124.76,
          "close": 125.33,
          "volume": 37836719
        },
        {
          "time": "2026-09-30",
          "open": 124.47,
          "high": 125,
          "low": 124.02,
          "close": 124.55,
          "volume": 20027951
        },
        {
          "time": "2026-10-01",
          "open": 123.69,
          "high": 125.24,
          "low": 122.86,
          "close": 124.41,
          "volume": 36021413
        },
        {
          "time": "2026-10-02",
          "open": 123.54,
          "high": 126.33,
          "low": 122.24,
          "close": 125.03,
          "volume": 49708842
        },
        {
          "time": "2026-10-05",
          "open": 124.17,
          "high": 128.14,
          "low": 122.43,
          "close": 126.4,
          "volume": 30188280
        }
      ],
      "ema20": [
        {
          "time": "2026-08-11",
          "value": 116.36
        },
        {
          "time": "2026-08-12",
          "value": 116.42
        },
        {
          "time": "2026-08-13",
          "value": 116.58
        },
        {
          "time": "2026-08-14",
          "value": 116.85
        },
        {
          "time": "2026-08-17",
          "value": 117.21
        },
        {
          "time": "2026-08-18",
          "value": 117.62
        },
        {
          "time": "2026-08-19",
          "value": 118.02
        },
        {
          "time": "2026-08-20",
          "value": 118.35
        },
        {
          "time": "2026-08-21",
          "value": 118.58
        },
        {
          "time": "2026-08-24",
          "value": 118.68
        },
        {
          "time": "2026-08-25",
          "value": 118.67
        },
        {
          "time": "2026-08-26",
          "value": 118.6
        },
        {
          "time": "2026-08-27",
          "value": 118.51
        },
        {
          "time": "2026-08-28",
          "value": 118.47
        },
        {
          "time": "2026-08-31",
          "value": 118.53
        },
        {
          "time": "2026-09-01",
          "value": 118.72
        },
        {
          "time": "2026-09-02",
          "value": 119.05
        },
        {
          "time": "2026-09-03",
          "value": 119.47
        },
        {
          "time": "2026-09-04",
          "value": 119.92
        },
        {
          "time": "2026-09-07",
          "value": 120.35
        },
        {
          "time": "2026-09-08",
          "value": 120.69
        },
        {
          "time": "2026-09-09",
          "value": 120.9
        },
        {
          "time": "2026-09-10",
          "value": 120.99
        },
        {
          "time": "2026-09-11",
          "value": 120.97
        },
        {
          "time": "2026-09-14",
          "value": 120.91
        },
        {
          "time": "2026-09-15",
          "value": 120.87
        },
        {
          "time": "2026-09-16",
          "value": 120.92
        },
        {
          "time": "2026-09-17",
          "value": 121.1
        },
        {
          "time": "2026-09-18",
          "value": 121.44
        },
        {
          "time": "2026-09-21",
          "value": 121.9
        },
        {
          "time": "2026-09-22",
          "value": 122.45
        },
        {
          "time": "2026-09-23",
          "value": 123.02
        },
        {
          "time": "2026-09-24",
          "value": 123.52
        },
        {
          "time": "2026-09-25",
          "value": 123.91
        },
        {
          "time": "2026-09-28",
          "value": 124.15
        },
        {
          "time": "2026-09-29",
          "value": 124.27
        },
        {
          "time": "2026-09-30",
          "value": 124.29
        },
        {
          "time": "2026-10-01",
          "value": 124.3
        },
        {
          "time": "2026-10-02",
          "value": 124.37
        },
        {
          "time": "2026-10-05",
          "value": 124.57
        }
      ],
      "chartMarkers": [
        {
          "time": "2026-09-15",
          "position": "aboveBar",
          "color": "#f59e0b",
          "shape": "arrowUp",
          "text": "CONGRESS: Rep. Pelosi BUY ($1M-$5M)"
        },
        {
          "time": "2026-09-28",
          "position": "belowBar",
          "color": "#10b981",
          "shape": "circle",
          "text": "FORM 4: J. Huang Option Exercise"
        }
      ],
      "corporateEvent": {
        "coverage_status": "CONFIRMED_UPCOMING",
        "preflight_state": "WARN",
        "freshness_status": "LIVE",
        "days_to_earnings": 3.0,
        "has_imminent_earnings": true
      },
      "preflight": {
        "isApplicable": true,
        "isCleared": false,
        "passedCount": 4,
        "totalMandatory": 5,
        "activeBlocker": "Imminent Earnings Hazard (3.0 Days to Binary Earnings)",
        "whatNeedsToChange": "Wait for post-earnings binary risk window to clear (>7 days).",
        "evidenceFreshness": "LIVE",
        "evidenceCompleteness": "COMPLETE",
        "checks": [
          {
            "id": "CHK-1",
            "name": "Reward/Risk Asymmetry",
            "rule": "Reward:Risk >= 2.0:1",
            "status": "PASS",
            "badge": "PASS",
            "value": "2.0:1",
            "authority": "Canonical Decision Hierarchy (Trade Sizing Engine)",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Target 1 ($136.20) to Invalidation ($121.50) satisfies 2.0:1 ratio.",
            "recoveryAction": "Awaiting price retest within optimal accumulation buy zone."
          },
          {
            "id": "CHK-2",
            "name": "Technical Structure",
            "rule": "Stage 2 Base Breakout (Not Stage 4)",
            "status": "PASS",
            "badge": "PASS",
            "value": "Stage 2 VCP (Confirmed)",
            "authority": "Minervini Volatility Contraction Pattern Analyzer",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Valid 3-stage VCP breakout above $125.80 pivot.",
            "recoveryAction": "Wait for primary base consolidation to establish higher low floor."
          },
          {
            "id": "CHK-3",
            "name": "Capital Risk",
            "rule": "Short Float < 12% & Quality Score >= 60",
            "status": "PASS",
            "badge": "PASS",
            "value": "Short Float: 3.2% | Quality: 92/100",
            "authority": "Master Asset Catalog & Fundamental Solvency Ledger",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Low short float (3.2%) and top-tier quality score (92).",
            "recoveryAction": "Resolve incomplete balance sheet or short interest reporting."
          },
          {
            "id": "CHK-4",
            "name": "Catalyst Hazard",
            "rule": ">7 Days to Forward Binary Event",
            "status": "HAZARD",
            "badge": "HAZARD",
            "value": "3.0 Days to Binary Earnings (Confirmed)",
            "authority": "SEC EDGAR Form 8-K / Corporate Event Schedule",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "\u26a0\ufe0f Imminent Binary Event: Overnight gap risk exceeds standard stop constraint (3.0 days <= 7.0d threshold).",
            "recoveryAction": "Wait for post-earnings binary risk window to clear (>7 days)."
          },
          {
            "id": "CHK-5",
            "name": "Macro Regime",
            "rule": "CBOE VIX Index < 26.0",
            "status": "PASS",
            "badge": "PASS",
            "value": "VIX 19.5",
            "authority": "CBOE Volatility Index & Canonical Macro Ribbon",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Macro regime stable (VIX 19.5 < 26.0 threshold).",
            "recoveryAction": "Wait for market-wide volatility index (VIX) to cool below 26.0 risk gate."
          }
        ]
      }
    },
    "AMD": {
      "state_id": "STATE_2_WAITING_PULLBACK",
      "symbol": "AMD",
      "name": "Advanced Micro Devices",
      "exchange": "NASDAQ",
      "sector": "Large Cap Semiconductors",
      "volumeStatement": "38.1M (112% vs 50D avg)",
      "currentPrice": 164.2,
      "priceDeltaDollar": 3.45,
      "priceDeltaPct": 2.15,
      "decisionState": "VALID_SETUP",
      "executionStatus": "WAITING_FOR_TRIGGER",
      "isActionable": false,
      "pattern": "Base 1 Pivot Cleared \u2014 Pullback Pending",
      "confluenceScore": 74,
      "confluenceRating": "CONSTRUCTIVE ACCUMULATION CONVICTION",
      "confluenceBadge": "3-Feed Confluence (Selective)",
      "levels": {
        "pivot": 162,
        "entryMin": 160,
        "entryMax": 162.5,
        "referencePrice": 164.2,
        "stopLoss": 154,
        "stopDistanceDollar": 10.2,
        "stopDistancePct": -6.21,
        "target1": 182,
        "rewardDistanceDollar": 17.8,
        "rewardDistancePct": 10.84,
        "rewardRiskRatio": 1.75,
        "riskRewardDisplay": "1 : 1.75 (Sub-threshold)",
        "rMultipleTarget1": 1.75
      },
      "sizing": {
        "shares": 0,
        "capitalAllocated": 0,
        "portfolioAllocationPct": 0,
        "modeledStopLossDollar": 0,
        "modeledStopLossEquityPct": 0,
        "riskBudgetDollar": 2490,
        "riskBudgetUtilizationPct": 0,
        "bindingConstraint": "Awaiting Price Pullback (Entry outside optimal buy zone)",
        "disclaimer": "Non-actionable: R:R is 1:1.75 at market quote $164.20. Requires pullback to $160.00\u2013$162.50 to clear minimum 2.0R gate."
      },
      "pillars": [
        {
          "pillar": "TECHNICAL_STRUCTURE",
          "label": "Technical Confluence",
          "weightPct": 25,
          "score": 75,
          "status": "neutral",
          "detail": "Extended +1.4% above optimal pivot. Awaiting orderly test of 10-day moving average."
        },
        {
          "pillar": "FUNDAMENTAL_SOLVENCY",
          "label": "Fundamental Moat",
          "weightPct": 25,
          "score": 82,
          "status": "positive",
          "detail": "Piotroski F-Score 8/9; data center revenue growing +115% YoY."
        },
        {
          "pillar": "SMART_MONEY_FLOW",
          "label": "Institutional Flow",
          "weightPct": 25,
          "score": 78,
          "status": "positive",
          "detail": "Moderate institutional accumulation (+12M USD net block order inflow)."
        },
        {
          "pillar": "MACRO_SAFETY_FLOOR",
          "label": "Macro Environment",
          "weightPct": 25,
          "score": 77,
          "status": "positive",
          "detail": "Broad semiconductor sector tailwinds supportive."
        }
      ],
      "provenance": {
        "asOf": "2026-10-06 (Static Baseline Fixture)",
        "authority": "ARX Ephemeral Prototype Fixture Authority (Offline Simulation)",
        "quoteFreshness": "STATIC_OFFLINE_PROTOTYPE",
        "dataMode": "ILLUSTRATIVE_SYNTHETIC_FIXTURE",
        "marketDataFreshness": "NOT_APPLICABLE_TO_STATIC_PROTOTYPE"
      },
      "secType": "COMMON_STOCK",
      "security_type": "COMMON_STOCK",
      "display_mode": "SINGLE_STOCK_ANALYSIS",
      "executionEligibility": "STOCK_EXECUTION",
      "candles": [
        {
          "time": "2026-08-11",
          "open": 167.77,
          "high": 168.52,
          "low": 166.6,
          "close": 167.35,
          "volume": 35000000
        },
        {
          "time": "2026-08-12",
          "open": 167.64,
          "high": 169.07,
          "low": 166.79,
          "close": 168.22,
          "volume": 49874972
        },
        {
          "time": "2026-08-13",
          "open": 168.51,
          "high": 171.16,
          "low": 167.2,
          "close": 169.85,
          "volume": 31166883
        },
        {
          "time": "2026-08-14",
          "open": 170.14,
          "high": 173.33,
          "low": 168.63,
          "close": 171.82,
          "volume": 21112780
        },
        {
          "time": "2026-08-17",
          "open": 172.11,
          "high": 175.03,
          "low": 170.7,
          "close": 173.62,
          "volume": 42411700
        },
        {
          "time": "2026-08-18",
          "open": 173.91,
          "high": 175.78,
          "low": 172.9,
          "close": 174.77,
          "volume": 46977307
        },
        {
          "time": "2026-08-19",
          "open": 175.06,
          "high": 175.64,
          "low": 174.35,
          "close": 174.93,
          "volume": 24501880
        },
        {
          "time": "2026-08-20",
          "open": 175.22,
          "high": 176.43,
          "low": 172.82,
          "close": 174.03,
          "volume": 25727943
        },
        {
          "time": "2026-08-21",
          "open": 174.32,
          "high": 176.05,
          "low": 170.54,
          "close": 172.27,
          "volume": 47887427
        },
        {
          "time": "2026-08-24",
          "open": 172.56,
          "high": 174.56,
          "low": 168.06,
          "close": 170.06,
          "volume": 40951109
        },
        {
          "time": "2026-08-25",
          "open": 170.35,
          "high": 172.3,
          "low": 165.98,
          "close": 167.93,
          "volume": 20579038
        },
        {
          "time": "2026-08-26",
          "open": 168.22,
          "high": 169.84,
          "low": 164.74,
          "close": 166.36,
          "volume": 32765015
        },
        {
          "time": "2026-08-27",
          "open": 166.65,
          "high": 167.74,
          "low": 164.58,
          "close": 165.67,
          "volume": 49996894
        },
        {
          "time": "2026-08-28",
          "open": 165.96,
          "high": 166.51,
          "low": 165.44,
          "close": 165.99,
          "volume": 33370451
        },
        {
          "time": "2026-08-31",
          "open": 166.28,
          "high": 168.25,
          "low": 165.23,
          "close": 167.2,
          "volume": 20423023
        },
        {
          "time": "2026-09-01",
          "open": 167.49,
          "high": 170.39,
          "low": 166.09,
          "close": 168.99,
          "volume": 40385875
        },
        {
          "time": "2026-09-02",
          "open": 169.28,
          "high": 172.37,
          "low": 167.81,
          "close": 170.9,
          "volume": 48189096
        },
        {
          "time": "2026-09-03",
          "open": 171.19,
          "high": 173.67,
          "low": 169.95,
          "close": 172.43,
          "volume": 26215440
        },
        {
          "time": "2026-09-04",
          "open": 172.72,
          "high": 173.93,
          "low": 171.95,
          "close": 173.16,
          "volume": 24074588
        },
        {
          "time": "2026-09-07",
          "open": 173.45,
          "high": 174.3,
          "low": 172.02,
          "close": 172.86,
          "volume": 46599918
        },
        {
          "time": "2026-09-08",
          "open": 173.15,
          "high": 174.58,
          "low": 170.17,
          "close": 171.6,
          "volume": 42936240
        },
        {
          "time": "2026-09-09",
          "open": 171.89,
          "high": 173.73,
          "low": 167.82,
          "close": 169.66,
          "volume": 21355000
        },
        {
          "time": "2026-09-10",
          "open": 169.95,
          "high": 171.92,
          "low": 165.53,
          "close": 167.5,
          "volume": 30579926
        },
        {
          "time": "2026-09-11",
          "open": 167.79,
          "high": 169.59,
          "low": 163.83,
          "close": 165.63,
          "volume": 49784004
        },
        {
          "time": "2026-09-14",
          "open": 165.92,
          "high": 167.3,
          "low": 163.08,
          "close": 164.46,
          "volume": 35610399
        },
        {
          "time": "2026-09-15",
          "open": 164.75,
          "high": 165.56,
          "low": 163.42,
          "close": 164.23,
          "volume": 20058702
        },
        {
          "time": "2026-09-16",
          "open": 164.52,
          "high": 165.71,
          "low": 163.76,
          "close": 164.95,
          "volume": 38239809
        },
        {
          "time": "2026-09-17",
          "open": 165.24,
          "high": 167.64,
          "low": 164.03,
          "close": 166.43,
          "volume": 49106434
        },
        {
          "time": "2026-09-18",
          "open": 166.72,
          "high": 169.72,
          "low": 165.28,
          "close": 168.28,
          "volume": 28125118
        },
        {
          "time": "2026-09-21",
          "open": 168.57,
          "high": 171.42,
          "low": 167.19,
          "close": 170.04,
          "volume": 22665147
        },
        {
          "time": "2026-09-22",
          "open": 170.33,
          "high": 172.28,
          "low": 169.29,
          "close": 171.24,
          "volume": 45053438
        },
        {
          "time": "2026-09-23",
          "open": 171.53,
          "high": 172.05,
          "low": 171.02,
          "close": 171.54,
          "volume": 44744193
        },
        {
          "time": "2026-09-24",
          "open": 171.83,
          "high": 172.92,
          "low": 169.75,
          "close": 170.84,
          "volume": 22435591
        },
        {
          "time": "2026-09-25",
          "open": 171.13,
          "high": 172.73,
          "low": 167.69,
          "close": 169.29,
          "volume": 28493517
        },
        {
          "time": "2026-09-28",
          "open": 169.58,
          "high": 171.47,
          "low": 165.38,
          "close": 167.27,
          "volume": 49241058
        },
        {
          "time": "2026-09-29",
          "open": 167.56,
          "high": 169.45,
          "low": 163.36,
          "close": 165.25,
          "volume": 37836719
        },
        {
          "time": "2026-09-30",
          "open": 165.54,
          "high": 167.14,
          "low": 162.1,
          "close": 163.7,
          "volume": 20027951
        },
        {
          "time": "2026-10-01",
          "open": 163.99,
          "high": 165.11,
          "low": 161.83,
          "close": 162.95,
          "volume": 36021413
        },
        {
          "time": "2026-10-02",
          "open": 163.24,
          "high": 163.8,
          "low": 162.58,
          "close": 163.14,
          "volume": 49708842
        },
        {
          "time": "2026-10-05",
          "open": 163.43,
          "high": 165.16,
          "low": 162.47,
          "close": 164.2,
          "volume": 30188280
        }
      ],
      "ema20": [
        {
          "time": "2026-08-11",
          "value": 167.35
        },
        {
          "time": "2026-08-12",
          "value": 167.43
        },
        {
          "time": "2026-08-13",
          "value": 167.66
        },
        {
          "time": "2026-08-14",
          "value": 168.06
        },
        {
          "time": "2026-08-17",
          "value": 168.59
        },
        {
          "time": "2026-08-18",
          "value": 169.18
        },
        {
          "time": "2026-08-19",
          "value": 169.73
        },
        {
          "time": "2026-08-20",
          "value": 170.14
        },
        {
          "time": "2026-08-21",
          "value": 170.34
        },
        {
          "time": "2026-08-24",
          "value": 170.31
        },
        {
          "time": "2026-08-25",
          "value": 170.09
        },
        {
          "time": "2026-08-26",
          "value": 169.73
        },
        {
          "time": "2026-08-27",
          "value": 169.34
        },
        {
          "time": "2026-08-28",
          "value": 169.02
        },
        {
          "time": "2026-08-31",
          "value": 168.85
        },
        {
          "time": "2026-09-01",
          "value": 168.86
        },
        {
          "time": "2026-09-02",
          "value": 169.06
        },
        {
          "time": "2026-09-03",
          "value": 169.38
        },
        {
          "time": "2026-09-04",
          "value": 169.74
        },
        {
          "time": "2026-09-07",
          "value": 170.04
        },
        {
          "time": "2026-09-08",
          "value": 170.19
        },
        {
          "time": "2026-09-09",
          "value": 170.14
        },
        {
          "time": "2026-09-10",
          "value": 169.88
        },
        {
          "time": "2026-09-11",
          "value": 169.48
        },
        {
          "time": "2026-09-14",
          "value": 169
        },
        {
          "time": "2026-09-15",
          "value": 168.55
        },
        {
          "time": "2026-09-16",
          "value": 168.2
        },
        {
          "time": "2026-09-17",
          "value": 168.04
        },
        {
          "time": "2026-09-18",
          "value": 168.06
        },
        {
          "time": "2026-09-21",
          "value": 168.25
        },
        {
          "time": "2026-09-22",
          "value": 168.53
        },
        {
          "time": "2026-09-23",
          "value": 168.82
        },
        {
          "time": "2026-09-24",
          "value": 169.01
        },
        {
          "time": "2026-09-25",
          "value": 169.04
        },
        {
          "time": "2026-09-28",
          "value": 168.87
        },
        {
          "time": "2026-09-29",
          "value": 168.52
        },
        {
          "time": "2026-09-30",
          "value": 168.07
        },
        {
          "time": "2026-10-01",
          "value": 167.58
        },
        {
          "time": "2026-10-02",
          "value": 167.16
        },
        {
          "time": "2026-10-05",
          "value": 166.87
        }
      ],
      "chartMarkers": [
        {
          "time": "2026-09-12",
          "position": "aboveBar",
          "color": "#f59e0b",
          "shape": "arrowUp",
          "text": "CONGRESS: Sen. Mullin BUY ($100k-$250k)"
        }
      ],
      "corporateEvent": {
        "coverage_status": "CONFIRMED_UPCOMING",
        "preflight_state": "PASS",
        "freshness_status": "LIVE",
        "days_to_earnings": 21.0,
        "has_imminent_earnings": false
      },
      "preflight": {
        "isApplicable": true,
        "isCleared": false,
        "passedCount": 5,
        "totalMandatory": 5,
        "activeBlocker": "Canonical Setup Non-Actionable (Awaiting Breakout Trigger)",
        "whatNeedsToChange": "Wait for breakout confirmation trigger above pivot on volume.",
        "evidenceFreshness": "LIVE",
        "evidenceCompleteness": "COMPLETE",
        "checks": [
          {
            "id": "CHK-1",
            "name": "Reward/Risk Asymmetry",
            "rule": "Reward:Risk >= 2.0:1",
            "status": "PASS",
            "badge": "PASS",
            "value": "2.1:1",
            "authority": "Canonical Decision Hierarchy (Trade Sizing Engine)",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "R:R 2.1:1 satisfies minimum asymmetric criteria.",
            "recoveryAction": "Awaiting price retest within optimal accumulation buy zone."
          },
          {
            "id": "CHK-2",
            "name": "Technical Structure",
            "rule": "Stage 2 Base Breakout (Not Stage 4)",
            "status": "PASS",
            "badge": "PASS",
            "value": "Stage 2 VCP (Forming)",
            "authority": "Minervini Volatility Contraction Pattern Analyzer",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Valid base structure forming, but breakout trigger not yet confirmed.",
            "recoveryAction": "Wait for breakout confirmation trigger above pivot on volume."
          },
          {
            "id": "CHK-3",
            "name": "Capital Risk",
            "rule": "Short Float < 12% & Quality Score >= 60",
            "status": "PASS",
            "badge": "PASS",
            "value": "Short Float: 4.8% | Quality: 80/100",
            "authority": "Master Asset Catalog & Fundamental Solvency Ledger",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Short interest and solvency ratings within safe operating bands.",
            "recoveryAction": "Resolve incomplete balance sheet or short interest reporting."
          },
          {
            "id": "CHK-4",
            "name": "Catalyst Hazard",
            "rule": ">7 Days to Forward Binary Event",
            "status": "PASS",
            "badge": "PASS",
            "value": "21.0 Days to Binary Earnings (Confirmed)",
            "authority": "SEC EDGAR Corporate Events Authority",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Clean 21.0-day forward earnings buffer.",
            "recoveryAction": "Wait for binary catalyst risk event to pass (>7 days buffer required)."
          },
          {
            "id": "CHK-5",
            "name": "Macro Regime",
            "rule": "CBOE VIX Index < 26.0",
            "status": "PASS",
            "badge": "PASS",
            "value": "VIX 18.0",
            "authority": "CBOE Volatility Index & Canonical Macro Ribbon",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Macro regime supportive (VIX 18.0 < 26.0).",
            "recoveryAction": "Wait for market-wide volatility index (VIX) to cool below 26.0 risk gate."
          }
        ]
      }
    },
    "PLSE": {
      "state_id": "STATE_3_STALE_EVIDENCE",
      "symbol": "PLSE",
      "name": "Pulse Biosciences, Inc.",
      "exchange": "NASDAQ",
      "sector": "Medical Devices",
      "volumeStatement": "0.4M (Unverified / Stale)",
      "currentPrice": 18.9,
      "priceDeltaDollar": -0.3,
      "priceDeltaPct": -1.56,
      "decisionState": "EVIDENCE_INCOMPLETE",
      "executionStatus": "STALE_MARKET_DATA",
      "isActionable": false,
      "pattern": "Base Consolidation (Unverified)",
      "confluenceScore": null,
      "confluenceRating": "EVIDENCE INCOMPLETE \u2014 STALE FEED",
      "confluenceBadge": "Feed Disconnected",
      "levels": {
        "pivot": 0,
        "entryMin": 0,
        "entryMax": 0,
        "referencePrice": 18.9,
        "stopLoss": 0,
        "stopDistanceDollar": 0,
        "stopDistancePct": 0,
        "target1": 0,
        "rewardDistanceDollar": 0,
        "rewardDistancePct": 0,
        "rewardRiskRatio": 0,
        "riskRewardDisplay": "Unavailable",
        "rMultipleTarget1": 0
      },
      "sizing": {
        "shares": 0,
        "capitalAllocated": 0,
        "portfolioAllocationPct": 0,
        "modeledStopLossDollar": 0,
        "modeledStopLossEquityPct": 0,
        "riskBudgetDollar": 2490,
        "riskBudgetUtilizationPct": 0,
        "bindingConstraint": "Analytical Feed Offline (>24h Staleness)",
        "disclaimer": "Sizing disabled: Analytical backend engine unreachable. Point-in-time regulatory filings unverified."
      },
      "pillars": [
        {
          "pillar": "TECHNICAL_STRUCTURE",
          "label": "Technical Confluence",
          "weightPct": 25,
          "score": 0,
          "status": "unavailable",
          "detail": "Market data stale >24 hours. Awaiting tape synchronization."
        },
        {
          "pillar": "FUNDAMENTAL_SOLVENCY",
          "label": "Fundamental Moat",
          "weightPct": 25,
          "score": 0,
          "status": "unavailable",
          "detail": "SEC 10-Q filing ingestion incomplete."
        },
        {
          "pillar": "SMART_MONEY_FLOW",
          "label": "Institutional Flow",
          "weightPct": 25,
          "score": 0,
          "status": "unavailable",
          "detail": "Insider filing stream disconnected."
        },
        {
          "pillar": "MACRO_SAFETY_FLOOR",
          "label": "Macro Environment",
          "weightPct": 25,
          "score": 0,
          "status": "unavailable",
          "detail": "Telemetry unavailable."
        }
      ],
      "provenance": {
        "asOf": "2026-10-06 (Static Baseline Fixture)",
        "authority": "ARX Ephemeral Prototype Fixture Authority (Offline Simulation)",
        "quoteFreshness": "STATIC_OFFLINE_PROTOTYPE",
        "dataMode": "ILLUSTRATIVE_SYNTHETIC_FIXTURE",
        "marketDataFreshness": "NOT_APPLICABLE_TO_STATIC_PROTOTYPE"
      },
      "secType": "COMMON_STOCK",
      "security_type": "COMMON_STOCK",
      "display_mode": "SINGLE_STOCK_ANALYSIS",
      "executionEligibility": "STOCK_EXECUTION",
      "candles": [
        {
          "time": "2026-08-11",
          "open": 18.47,
          "high": 19.16,
          "low": 17.84,
          "close": 18.53,
          "volume": 35000000
        },
        {
          "time": "2026-08-12",
          "open": 18.47,
          "high": 19.37,
          "low": 17.76,
          "close": 18.66,
          "volume": 49874972
        },
        {
          "time": "2026-08-13",
          "open": 18.61,
          "high": 19.68,
          "low": 17.83,
          "close": 18.9,
          "volume": 31166883
        },
        {
          "time": "2026-08-14",
          "open": 18.85,
          "high": 20,
          "low": 18.04,
          "close": 19.19,
          "volume": 21112780
        },
        {
          "time": "2026-08-17",
          "open": 19.13,
          "high": 20.24,
          "low": 18.34,
          "close": 19.45,
          "volume": 42411700
        },
        {
          "time": "2026-08-18",
          "open": 19.4,
          "high": 20.38,
          "low": 18.66,
          "close": 19.64,
          "volume": 46977307
        },
        {
          "time": "2026-08-19",
          "open": 19.59,
          "high": 20.35,
          "low": 18.93,
          "close": 19.69,
          "volume": 24501880
        },
        {
          "time": "2026-08-20",
          "open": 19.63,
          "high": 20.26,
          "low": 18.95,
          "close": 19.58,
          "volume": 25727943
        },
        {
          "time": "2026-08-21",
          "open": 19.53,
          "high": 20.23,
          "low": 18.66,
          "close": 19.36,
          "volume": 47887427
        },
        {
          "time": "2026-08-24",
          "open": 19.31,
          "high": 20.04,
          "low": 18.34,
          "close": 19.08,
          "volume": 40951109
        },
        {
          "time": "2026-08-25",
          "open": 19.02,
          "high": 19.75,
          "low": 18.07,
          "close": 18.8,
          "volume": 20579038
        },
        {
          "time": "2026-08-26",
          "open": 18.75,
          "high": 19.44,
          "low": 17.91,
          "close": 18.6,
          "volume": 32765015
        },
        {
          "time": "2026-08-27",
          "open": 18.54,
          "high": 19.15,
          "low": 17.91,
          "close": 18.52,
          "volume": 49996894
        },
        {
          "time": "2026-08-28",
          "open": 18.47,
          "high": 19.26,
          "low": 17.8,
          "close": 18.59,
          "volume": 33370451
        },
        {
          "time": "2026-08-31",
          "open": 18.54,
          "high": 19.54,
          "low": 17.79,
          "close": 18.79,
          "volume": 20423023
        },
        {
          "time": "2026-09-01",
          "open": 18.73,
          "high": 19.86,
          "low": 17.93,
          "close": 19.06,
          "volume": 40385875
        },
        {
          "time": "2026-09-02",
          "open": 19.01,
          "high": 20.18,
          "low": 18.2,
          "close": 19.37,
          "volume": 48189096
        },
        {
          "time": "2026-09-03",
          "open": 19.32,
          "high": 20.4,
          "low": 18.54,
          "close": 19.62,
          "volume": 26215440
        },
        {
          "time": "2026-09-04",
          "open": 19.56,
          "high": 20.46,
          "low": 18.85,
          "close": 19.75,
          "volume": 24074588
        },
        {
          "time": "2026-09-07",
          "open": 19.7,
          "high": 20.36,
          "low": 19.08,
          "close": 19.74,
          "volume": 46599918
        },
        {
          "time": "2026-09-08",
          "open": 19.69,
          "high": 20.35,
          "low": 18.92,
          "close": 19.58,
          "volume": 42936240
        },
        {
          "time": "2026-09-09",
          "open": 19.52,
          "high": 20.25,
          "low": 18.58,
          "close": 19.31,
          "volume": 21355000
        },
        {
          "time": "2026-09-10",
          "open": 19.26,
          "high": 20.01,
          "low": 18.26,
          "close": 19.01,
          "volume": 30579926
        },
        {
          "time": "2026-09-11",
          "open": 18.96,
          "high": 19.68,
          "low": 18.03,
          "close": 18.75,
          "volume": 49784004
        },
        {
          "time": "2026-09-14",
          "open": 18.69,
          "high": 19.35,
          "low": 17.93,
          "close": 18.59,
          "volume": 35610399
        },
        {
          "time": "2026-09-15",
          "open": 18.54,
          "high": 19.2,
          "low": 17.92,
          "close": 18.58,
          "volume": 20058702
        },
        {
          "time": "2026-09-16",
          "open": 18.53,
          "high": 19.43,
          "low": 17.82,
          "close": 18.72,
          "volume": 38239809
        },
        {
          "time": "2026-09-17",
          "open": 18.66,
          "high": 19.74,
          "low": 17.88,
          "close": 18.96,
          "volume": 49106434
        },
        {
          "time": "2026-09-18",
          "open": 18.91,
          "high": 20.1,
          "low": 18.09,
          "close": 19.28,
          "volume": 28125118
        },
        {
          "time": "2026-09-21",
          "open": 19.22,
          "high": 20.38,
          "low": 18.42,
          "close": 19.58,
          "volume": 22665147
        },
        {
          "time": "2026-09-22",
          "open": 19.52,
          "high": 20.55,
          "low": 18.76,
          "close": 19.79,
          "volume": 45053438
        },
        {
          "time": "2026-09-23",
          "open": 19.74,
          "high": 20.55,
          "low": 19.06,
          "close": 19.87,
          "volume": 44744193
        },
        {
          "time": "2026-09-24",
          "open": 19.81,
          "high": 20.43,
          "low": 19.16,
          "close": 19.78,
          "volume": 22435591
        },
        {
          "time": "2026-09-25",
          "open": 19.73,
          "high": 20.43,
          "low": 18.86,
          "close": 19.56,
          "volume": 28493517
        },
        {
          "time": "2026-09-28",
          "open": 19.51,
          "high": 20.26,
          "low": 18.52,
          "close": 19.27,
          "volume": 49241058
        },
        {
          "time": "2026-09-29",
          "open": 19.21,
          "high": 19.96,
          "low": 18.21,
          "close": 18.96,
          "volume": 37836719
        },
        {
          "time": "2026-09-30",
          "open": 18.91,
          "high": 19.61,
          "low": 18.04,
          "close": 18.74,
          "volume": 20027951
        },
        {
          "time": "2026-10-01",
          "open": 18.69,
          "high": 19.32,
          "low": 18.02,
          "close": 18.65,
          "volume": 36021413
        },
        {
          "time": "2026-10-02",
          "open": 18.59,
          "high": 19.37,
          "low": 17.92,
          "close": 18.7,
          "volume": 49708842
        },
        {
          "time": "2026-10-05",
          "open": 18.65,
          "high": 19.65,
          "low": 17.9,
          "close": 18.9,
          "volume": 30188280
        }
      ],
      "ema20": [
        {
          "time": "2026-08-11",
          "value": 18.53
        },
        {
          "time": "2026-08-12",
          "value": 18.54
        },
        {
          "time": "2026-08-13",
          "value": 18.58
        },
        {
          "time": "2026-08-14",
          "value": 18.63
        },
        {
          "time": "2026-08-17",
          "value": 18.71
        },
        {
          "time": "2026-08-18",
          "value": 18.8
        },
        {
          "time": "2026-08-19",
          "value": 18.89
        },
        {
          "time": "2026-08-20",
          "value": 18.95
        },
        {
          "time": "2026-08-21",
          "value": 18.99
        },
        {
          "time": "2026-08-24",
          "value": 19
        },
        {
          "time": "2026-08-25",
          "value": 18.98
        },
        {
          "time": "2026-08-26",
          "value": 18.94
        },
        {
          "time": "2026-08-27",
          "value": 18.9
        },
        {
          "time": "2026-08-28",
          "value": 18.87
        },
        {
          "time": "2026-08-31",
          "value": 18.87
        },
        {
          "time": "2026-09-01",
          "value": 18.88
        },
        {
          "time": "2026-09-02",
          "value": 18.93
        },
        {
          "time": "2026-09-03",
          "value": 19
        },
        {
          "time": "2026-09-04",
          "value": 19.07
        },
        {
          "time": "2026-09-07",
          "value": 19.13
        },
        {
          "time": "2026-09-08",
          "value": 19.17
        },
        {
          "time": "2026-09-09",
          "value": 19.19
        },
        {
          "time": "2026-09-10",
          "value": 19.17
        },
        {
          "time": "2026-09-11",
          "value": 19.13
        },
        {
          "time": "2026-09-14",
          "value": 19.08
        },
        {
          "time": "2026-09-15",
          "value": 19.03
        },
        {
          "time": "2026-09-16",
          "value": 19
        },
        {
          "time": "2026-09-17",
          "value": 19
        },
        {
          "time": "2026-09-18",
          "value": 19.02
        },
        {
          "time": "2026-09-21",
          "value": 19.08
        },
        {
          "time": "2026-09-22",
          "value": 19.15
        },
        {
          "time": "2026-09-23",
          "value": 19.21
        },
        {
          "time": "2026-09-24",
          "value": 19.27
        },
        {
          "time": "2026-09-25",
          "value": 19.3
        },
        {
          "time": "2026-09-28",
          "value": 19.29
        },
        {
          "time": "2026-09-29",
          "value": 19.26
        },
        {
          "time": "2026-09-30",
          "value": 19.21
        },
        {
          "time": "2026-10-01",
          "value": 19.16
        },
        {
          "time": "2026-10-02",
          "value": 19.11
        },
        {
          "time": "2026-10-05",
          "value": 19.09
        }
      ],
      "chartMarkers": [],
      "preflight": {
        "isApplicable": true,
        "isCleared": false,
        "passedCount": 2,
        "totalMandatory": 5,
        "activeBlocker": "Capital Risk Evidence Incomplete (Quality Rating Missing)",
        "whatNeedsToChange": "Resolve incomplete short float or fundamental quality evidence.",
        "evidenceFreshness": "PARTIAL",
        "evidenceCompleteness": "PARTIAL",
        "checks": [
          {
            "id": "CHK-1",
            "name": "Reward/Risk Asymmetry",
            "rule": "Reward:Risk >= 2.0:1",
            "status": "PASS",
            "badge": "PASS",
            "value": "2.2:1",
            "authority": "Canonical Decision Hierarchy (Trade Sizing Engine)",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Theoretical R:R satisfies minimum ratio.",
            "recoveryAction": "Awaiting price retest within optimal accumulation buy zone."
          },
          {
            "id": "CHK-2",
            "name": "Technical Structure",
            "rule": "Stage 2 Base Breakout (Not Stage 4)",
            "status": "PASS",
            "badge": "PASS",
            "value": "Base Contraction",
            "authority": "Minervini Volatility Contraction Pattern Analyzer",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Base structure detected.",
            "recoveryAction": "Wait for primary base consolidation to establish higher low floor."
          },
          {
            "id": "CHK-3",
            "name": "Capital Risk",
            "rule": "Short Float < 12% & Quality Score >= 60",
            "status": "PARTIAL",
            "badge": "PARTIAL",
            "value": "Evidence Incomplete",
            "authority": "Master Asset Catalog & Fundamental Solvency Ledger",
            "timestamp": "2026-10-24 10:00 EDT - UNAVAILABLE",
            "explanation": "Evidence Incomplete: Fundamental quality score or short interest input missing.",
            "recoveryAction": "Resolve incomplete short float or fundamental quality evidence."
          },
          {
            "id": "CHK-4",
            "name": "Catalyst Hazard",
            "rule": ">7 Days to Forward Binary Event",
            "status": "UNAVAILABLE",
            "badge": "UNAVAILABLE",
            "value": "Coverage Unavailable",
            "authority": "SEC EDGAR Corporate Events Authority",
            "timestamp": "2026-10-24 10:00 EDT - UNAVAILABLE",
            "explanation": "Catalyst evidence unavailable: Forward earnings calendar unverified.",
            "recoveryAction": "Await verified forward corporate event and earnings calendar coverage."
          },
          {
            "id": "CHK-5",
            "name": "Macro Regime",
            "rule": "CBOE VIX Index < 26.0",
            "status": "PASS",
            "badge": "PASS",
            "value": "VIX 19.8",
            "authority": "CBOE Volatility Index & Canonical Macro Ribbon",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Macro regime supportive.",
            "recoveryAction": "Wait for market-wide volatility index (VIX) to cool below 26.0 risk gate."
          }
        ]
      }
    },
    "XYZ": {
      "state_id": "STATE_4_INELIGIBLE_ASSET",
      "symbol": "XYZ",
      "name": "Microcap Speculative Inc.",
      "exchange": "OTC",
      "sector": "Speculative Nano-Cap",
      "volumeStatement": "12.4K (Fails Liquidity Floor)",
      "currentPrice": 1.24,
      "priceDeltaDollar": -0.11,
      "priceDeltaPct": -8.15,
      "decisionState": "UNVERIFIED",
      "executionStatus": "UNVERIFIED_ASSET",
      "isActionable": false,
      "pattern": "Irregular Microcap Distribution",
      "confluenceScore": 0,
      "confluenceRating": "DEFENSIVE / CAPITAL PRESERVATION MODE",
      "confluenceBadge": "Prohibited / Ineligible",
      "levels": {
        "pivot": 0,
        "entryMin": 0,
        "entryMax": 0,
        "referencePrice": 1.24,
        "stopLoss": 0,
        "stopDistanceDollar": 0,
        "stopDistancePct": 0,
        "target1": 0,
        "rewardDistanceDollar": 0,
        "rewardDistancePct": 0,
        "rewardRiskRatio": 0,
        "riskRewardDisplay": "Ineligible",
        "rMultipleTarget1": 0
      },
      "sizing": {
        "shares": 0,
        "capitalAllocated": 0,
        "portfolioAllocationPct": 0,
        "modeledStopLossDollar": 0,
        "modeledStopLossEquityPct": 0,
        "riskBudgetDollar": 2490,
        "riskBudgetUtilizationPct": 0,
        "bindingConstraint": "Liquidity Gate Invariant Failure",
        "disclaimer": "Asset ineligible: 30-day average daily dollar volume <$10M USD. ARX institutional rules prohibit trading illiquid penny instruments."
      },
      "pillars": [
        {
          "pillar": "TECHNICAL_STRUCTURE",
          "label": "Technical Confluence",
          "weightPct": 25,
          "score": 15,
          "status": "warning",
          "detail": "Erratic spread and erratic price distribution."
        },
        {
          "pillar": "FUNDAMENTAL_SOLVENCY",
          "label": "Fundamental Moat",
          "weightPct": 25,
          "score": 10,
          "status": "warning",
          "detail": "Negative operating cash flow; ongoing equity dilution."
        },
        {
          "pillar": "SMART_MONEY_FLOW",
          "label": "Institutional Flow",
          "weightPct": 25,
          "score": 0,
          "status": "warning",
          "detail": "Zero institutional sponsorship."
        },
        {
          "pillar": "MACRO_SAFETY_FLOOR",
          "label": "Macro Environment",
          "weightPct": 25,
          "score": 0,
          "status": "warning",
          "detail": "Severe credit and liquidity friction."
        }
      ],
      "provenance": {
        "asOf": "2026-10-06 (Static Baseline Fixture)",
        "authority": "ARX Ephemeral Prototype Fixture Authority (Offline Simulation)",
        "quoteFreshness": "STATIC_OFFLINE_PROTOTYPE",
        "dataMode": "ILLUSTRATIVE_SYNTHETIC_FIXTURE",
        "marketDataFreshness": "NOT_APPLICABLE_TO_STATIC_PROTOTYPE"
      },
      "secType": "COMMON_STOCK",
      "security_type": "COMMON_STOCK",
      "display_mode": "SINGLE_STOCK_ANALYSIS",
      "executionEligibility": "FAIL_CLOSED",
      "candles": [
        {
          "time": "2026-08-11",
          "open": 1.41,
          "high": 2.43,
          "low": 0.35,
          "close": 1.37,
          "volume": 35000000
        },
        {
          "time": "2026-08-12",
          "open": 1.39,
          "high": 2.4,
          "low": 0.36,
          "close": 1.37,
          "volume": 49874972
        },
        {
          "time": "2026-08-13",
          "open": 1.39,
          "high": 2.4,
          "low": 0.37,
          "close": 1.38,
          "volume": 31166883
        },
        {
          "time": "2026-08-14",
          "open": 1.4,
          "high": 2.4,
          "low": 0.39,
          "close": 1.39,
          "volume": 21112780
        },
        {
          "time": "2026-08-17",
          "open": 1.41,
          "high": 2.42,
          "low": 0.39,
          "close": 1.4,
          "volume": 42411700
        },
        {
          "time": "2026-08-18",
          "open": 1.42,
          "high": 2.43,
          "low": 0.39,
          "close": 1.4,
          "volume": 46977307
        },
        {
          "time": "2026-08-19",
          "open": 1.43,
          "high": 2.45,
          "low": 0.38,
          "close": 1.4,
          "volume": 24501880
        },
        {
          "time": "2026-08-20",
          "open": 1.42,
          "high": 2.45,
          "low": 0.34,
          "close": 1.37,
          "volume": 25727943
        },
        {
          "time": "2026-08-21",
          "open": 1.39,
          "high": 2.43,
          "low": 0.29,
          "close": 1.33,
          "volume": 47887427
        },
        {
          "time": "2026-08-24",
          "open": 1.35,
          "high": 2.39,
          "low": 0.25,
          "close": 1.29,
          "volume": 40951109
        },
        {
          "time": "2026-08-25",
          "open": 1.31,
          "high": 2.34,
          "low": 0.22,
          "close": 1.25,
          "volume": 20579038
        },
        {
          "time": "2026-08-26",
          "open": 1.27,
          "high": 2.3,
          "low": 0.19,
          "close": 1.22,
          "volume": 32765015
        },
        {
          "time": "2026-08-27",
          "open": 1.24,
          "high": 2.26,
          "low": 0.18,
          "close": 1.2,
          "volume": 49996894
        },
        {
          "time": "2026-08-28",
          "open": 1.22,
          "high": 2.23,
          "low": 0.19,
          "close": 1.2,
          "volume": 33370451
        },
        {
          "time": "2026-08-31",
          "open": 1.22,
          "high": 2.23,
          "low": 0.2,
          "close": 1.21,
          "volume": 20423023
        },
        {
          "time": "2026-09-01",
          "open": 1.23,
          "high": 2.23,
          "low": 0.22,
          "close": 1.22,
          "volume": 40385875
        },
        {
          "time": "2026-09-02",
          "open": 1.24,
          "high": 2.24,
          "low": 0.24,
          "close": 1.24,
          "volume": 48189096
        },
        {
          "time": "2026-09-03",
          "open": 1.26,
          "high": 2.27,
          "low": 0.24,
          "close": 1.25,
          "volume": 26215440
        },
        {
          "time": "2026-09-04",
          "open": 1.27,
          "high": 2.28,
          "low": 0.24,
          "close": 1.25,
          "volume": 24074588
        },
        {
          "time": "2026-09-07",
          "open": 1.27,
          "high": 2.3,
          "low": 0.23,
          "close": 1.25,
          "volume": 46599918
        },
        {
          "time": "2026-09-08",
          "open": 1.27,
          "high": 2.29,
          "low": 0.21,
          "close": 1.23,
          "volume": 42936240
        },
        {
          "time": "2026-09-09",
          "open": 1.25,
          "high": 2.27,
          "low": 0.19,
          "close": 1.21,
          "volume": 21355000
        },
        {
          "time": "2026-09-10",
          "open": 1.23,
          "high": 2.25,
          "low": 0.17,
          "close": 1.19,
          "volume": 30579926
        },
        {
          "time": "2026-09-11",
          "open": 1.21,
          "high": 2.23,
          "low": 0.15,
          "close": 1.17,
          "volume": 49784004
        },
        {
          "time": "2026-09-14",
          "open": 1.19,
          "high": 2.21,
          "low": 0.14,
          "close": 1.16,
          "volume": 35610399
        },
        {
          "time": "2026-09-15",
          "open": 1.18,
          "high": 2.19,
          "low": 0.15,
          "close": 1.16,
          "volume": 20058702
        },
        {
          "time": "2026-09-16",
          "open": 1.18,
          "high": 2.19,
          "low": 0.16,
          "close": 1.17,
          "volume": 38239809
        },
        {
          "time": "2026-09-17",
          "open": 1.19,
          "high": 2.19,
          "low": 0.18,
          "close": 1.18,
          "volume": 49106434
        },
        {
          "time": "2026-09-18",
          "open": 1.2,
          "high": 2.2,
          "low": 0.2,
          "close": 1.2,
          "volume": 28125118
        },
        {
          "time": "2026-09-21",
          "open": 1.22,
          "high": 2.22,
          "low": 0.22,
          "close": 1.22,
          "volume": 22665147
        },
        {
          "time": "2026-09-22",
          "open": 1.24,
          "high": 2.24,
          "low": 0.23,
          "close": 1.23,
          "volume": 45053438
        },
        {
          "time": "2026-09-23",
          "open": 1.25,
          "high": 2.26,
          "low": 0.23,
          "close": 1.24,
          "volume": 44744193
        },
        {
          "time": "2026-09-24",
          "open": 1.26,
          "high": 2.27,
          "low": 0.23,
          "close": 1.24,
          "volume": 22435591
        },
        {
          "time": "2026-09-25",
          "open": 1.27,
          "high": 2.29,
          "low": 0.22,
          "close": 1.24,
          "volume": 28493517
        },
        {
          "time": "2026-09-28",
          "open": 1.26,
          "high": 2.28,
          "low": 0.21,
          "close": 1.23,
          "volume": 49241058
        },
        {
          "time": "2026-09-29",
          "open": 1.25,
          "high": 2.27,
          "low": 0.2,
          "close": 1.22,
          "volume": 37836719
        },
        {
          "time": "2026-09-30",
          "open": 1.24,
          "high": 2.25,
          "low": 0.21,
          "close": 1.22,
          "volume": 20027951
        },
        {
          "time": "2026-10-01",
          "open": 1.24,
          "high": 2.25,
          "low": 0.21,
          "close": 1.22,
          "volume": 36021413
        },
        {
          "time": "2026-10-02",
          "open": 1.24,
          "high": 2.25,
          "low": 0.22,
          "close": 1.23,
          "volume": 49708842
        },
        {
          "time": "2026-10-05",
          "open": 1.25,
          "high": 2.25,
          "low": 0.24,
          "close": 1.24,
          "volume": 30188280
        }
      ],
      "ema20": [
        {
          "time": "2026-08-11",
          "value": 1.37
        },
        {
          "time": "2026-08-12",
          "value": 1.37
        },
        {
          "time": "2026-08-13",
          "value": 1.37
        },
        {
          "time": "2026-08-14",
          "value": 1.37
        },
        {
          "time": "2026-08-17",
          "value": 1.38
        },
        {
          "time": "2026-08-18",
          "value": 1.38
        },
        {
          "time": "2026-08-19",
          "value": 1.38
        },
        {
          "time": "2026-08-20",
          "value": 1.38
        },
        {
          "time": "2026-08-21",
          "value": 1.37
        },
        {
          "time": "2026-08-24",
          "value": 1.37
        },
        {
          "time": "2026-08-25",
          "value": 1.36
        },
        {
          "time": "2026-08-26",
          "value": 1.34
        },
        {
          "time": "2026-08-27",
          "value": 1.33
        },
        {
          "time": "2026-08-28",
          "value": 1.32
        },
        {
          "time": "2026-08-31",
          "value": 1.31
        },
        {
          "time": "2026-09-01",
          "value": 1.3
        },
        {
          "time": "2026-09-02",
          "value": 1.29
        },
        {
          "time": "2026-09-03",
          "value": 1.29
        },
        {
          "time": "2026-09-04",
          "value": 1.28
        },
        {
          "time": "2026-09-07",
          "value": 1.28
        },
        {
          "time": "2026-09-08",
          "value": 1.28
        },
        {
          "time": "2026-09-09",
          "value": 1.27
        },
        {
          "time": "2026-09-10",
          "value": 1.26
        },
        {
          "time": "2026-09-11",
          "value": 1.25
        },
        {
          "time": "2026-09-14",
          "value": 1.24
        },
        {
          "time": "2026-09-15",
          "value": 1.24
        },
        {
          "time": "2026-09-16",
          "value": 1.23
        },
        {
          "time": "2026-09-17",
          "value": 1.23
        },
        {
          "time": "2026-09-18",
          "value": 1.22
        },
        {
          "time": "2026-09-21",
          "value": 1.22
        },
        {
          "time": "2026-09-22",
          "value": 1.22
        },
        {
          "time": "2026-09-23",
          "value": 1.23
        },
        {
          "time": "2026-09-24",
          "value": 1.23
        },
        {
          "time": "2026-09-25",
          "value": 1.23
        },
        {
          "time": "2026-09-28",
          "value": 1.23
        },
        {
          "time": "2026-09-29",
          "value": 1.23
        },
        {
          "time": "2026-09-30",
          "value": 1.23
        },
        {
          "time": "2026-10-01",
          "value": 1.23
        },
        {
          "time": "2026-10-02",
          "value": 1.23
        },
        {
          "time": "2026-10-05",
          "value": 1.23
        }
      ],
      "chartMarkers": [],
      "preflight": {
        "isApplicable": true,
        "isCleared": false,
        "passedCount": 0,
        "totalMandatory": 5,
        "activeBlocker": "Unverified Security (Fails Institutional Risk & Liquidity Floors)",
        "whatNeedsToChange": "Trading prohibited under ARX defensive capital preservation rules.",
        "evidenceFreshness": "UNVERIFIED",
        "evidenceCompleteness": "UNAVAILABLE",
        "checks": [
          {
            "id": "CHK-1",
            "name": "Reward/Risk Asymmetry",
            "rule": "Reward:Risk >= 2.0:1",
            "status": "FAIL",
            "badge": "FAIL",
            "value": "Unverified",
            "authority": "Canonical Decision Hierarchy",
            "timestamp": "UNAVAILABLE",
            "explanation": "Unverified spot price and execution levels.",
            "recoveryAction": "Awaiting verified exchange market feed."
          },
          {
            "id": "CHK-2",
            "name": "Technical Structure",
            "rule": "Stage 2 Base Breakout",
            "status": "FAIL",
            "badge": "FAIL",
            "value": "Unverified",
            "authority": "VCP Engine",
            "timestamp": "UNAVAILABLE",
            "explanation": "No verified base structure.",
            "recoveryAction": "Awaiting price history verification."
          },
          {
            "id": "CHK-3",
            "name": "Capital Risk",
            "rule": "Short Float < 12%",
            "status": "FAIL",
            "badge": "FAIL",
            "value": "Unverified",
            "authority": "Master Asset Catalog",
            "timestamp": "UNAVAILABLE",
            "explanation": "Microcap asset fails fundamental solvency floors.",
            "recoveryAction": "Security ineligible for institutional execution."
          },
          {
            "id": "CHK-4",
            "name": "Catalyst Hazard",
            "rule": ">7 Days to Binary Event",
            "status": "UNAVAILABLE",
            "badge": "UNAVAILABLE",
            "value": "Unverified",
            "authority": "SEC EDGAR",
            "timestamp": "UNAVAILABLE",
            "explanation": "No authoritative regulatory filing coverage.",
            "recoveryAction": "Trading prohibited."
          },
          {
            "id": "CHK-5",
            "name": "Macro Regime",
            "rule": "VIX < 26.0",
            "status": "FAIL",
            "badge": "FAIL",
            "value": "Unverified",
            "authority": "CBOE VIX",
            "timestamp": "UNAVAILABLE",
            "explanation": "Macro clearance withheld.",
            "recoveryAction": "Trading prohibited."
          }
        ]
      }
    },
    "AAPL": {
      "state_id": "STATE_5_ALREADY_HELD_MONITORING",
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "exchange": "NASDAQ",
      "sector": "Large Cap Consumer Tech",
      "volumeStatement": "42.5M (98% vs 50D avg)",
      "currentPrice": 224.3,
      "priceDeltaDollar": 1.9,
      "priceDeltaPct": 0.85,
      "decisionState": "ACTIONABLE_SETUP",
      "executionStatus": "IN_BUY_ZONE",
      "isActionable": true,
      "newEntryActionable": false,
      "uiPresentationMode": "POSITION_MONITORING",
      "positionMonitoringCanonical": false,
      "pattern": "Stage 2 Base Breakout",
      "confluenceScore": 86,
      "confluenceRating": "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT",
      "confluenceBadge": "4-Pillar Confluence (Pristine)",
      "levels": {
        "pivot": 220.0,
        "entryMin": 218.0,
        "entryMax": 222.0,
        "referencePrice": 224.3,
        "stopLoss": 218.5,
        "stopDistanceDollar": 5.8,
        "stopDistancePct": -2.59,
        "target1": 240.5,
        "rewardDistanceDollar": 16.2,
        "rewardDistancePct": 7.22,
        "rewardRiskRatio": 2.8,
        "riskRewardDisplay": "1 : 2.80",
        "rMultipleTarget1": 2.8
      },
      "sizing": {
        "shares": 100,
        "capitalAllocated": 22430.0,
        "portfolioAllocationPct": 18.02,
        "modeledStopLossDollar": 580.0,
        "modeledStopLossEquityPct": 0.47,
        "riskBudgetUtilizationPct": 23.29
      },
      "pillars": [
        {
          "pillar": "TECHNICAL_STRUCTURE",
          "label": "Technical Confluence",
          "weightPct": 25,
          "score": 86,
          "status": "positive",
          "detail": "Price advancing along 10 EMA; approaching Target 1 ($228.00)."
        },
        {
          "pillar": "FUNDAMENTAL_SOLVENCY",
          "label": "Fundamental Moat",
          "weightPct": 25,
          "score": 92,
          "status": "positive",
          "detail": "ROIC >50%; massive free cash flow generation; active share buyback."
        },
        {
          "pillar": "SMART_MONEY_FLOW",
          "label": "Institutional Flow",
          "weightPct": 25,
          "score": 80,
          "status": "positive",
          "detail": "Consistent institutional index and ETF rebalancing inflows."
        },
        {
          "pillar": "MACRO_SAFETY_FLOOR",
          "label": "Macro Environment",
          "weightPct": 25,
          "score": 78,
          "status": "positive",
          "detail": "Tech sector beta supportive."
        }
      ],
      "provenance": {
        "asOf": "2026-10-06 (Static Baseline Fixture)",
        "authority": "ARX Ephemeral Prototype Fixture Authority (Offline Simulation)",
        "quoteFreshness": "STATIC_OFFLINE_PROTOTYPE",
        "dataMode": "ILLUSTRATIVE_SYNTHETIC_FIXTURE",
        "marketDataFreshness": "NOT_APPLICABLE_TO_STATIC_PROTOTYPE"
      },
      "secType": "COMMON_STOCK",
      "security_type": "COMMON_STOCK",
      "display_mode": "SINGLE_STOCK_ANALYSIS",
      "executionEligibility": "STOCK_EXECUTION",
      "candles": [
        {
          "time": "2026-08-11",
          "open": 210.34,
          "high": 211.8,
          "low": 209.64,
          "close": 211.1,
          "volume": 35000000
        },
        {
          "time": "2026-08-12",
          "open": 210.6,
          "high": 213.03,
          "low": 209.54,
          "close": 211.97,
          "volume": 49874972
        },
        {
          "time": "2026-08-13",
          "open": 211.47,
          "high": 214.66,
          "low": 210.12,
          "close": 213.31,
          "volume": 31166883
        },
        {
          "time": "2026-08-14",
          "open": 212.81,
          "high": 216.35,
          "low": 211.33,
          "close": 214.87,
          "volume": 21112780
        },
        {
          "time": "2026-08-17",
          "open": 214.37,
          "high": 217.75,
          "low": 212.95,
          "close": 216.33,
          "volume": 42411700
        },
        {
          "time": "2026-08-18",
          "open": 215.83,
          "high": 218.58,
          "low": 214.65,
          "close": 217.4,
          "volume": 46977307
        },
        {
          "time": "2026-08-19",
          "open": 216.89,
          "high": 218.7,
          "low": 216.06,
          "close": 217.87,
          "volume": 24501880
        },
        {
          "time": "2026-08-20",
          "open": 217.37,
          "high": 218.14,
          "low": 216.93,
          "close": 217.7,
          "volume": 25727943
        },
        {
          "time": "2026-08-21",
          "open": 217.2,
          "high": 217.57,
          "low": 216.62,
          "close": 216.99,
          "volume": 47887427
        },
        {
          "time": "2026-08-24",
          "open": 216.49,
          "high": 217.03,
          "low": 215.45,
          "close": 215.99,
          "volume": 40951109
        },
        {
          "time": "2026-08-25",
          "open": 215.49,
          "high": 216.02,
          "low": 214.48,
          "close": 215.01,
          "volume": 20579038
        },
        {
          "time": "2026-08-26",
          "open": 214.51,
          "high": 214.83,
          "low": 214.05,
          "close": 214.37,
          "volume": 32765015
        },
        {
          "time": "2026-08-27",
          "open": 213.87,
          "high": 214.78,
          "low": 213.38,
          "close": 214.29,
          "volume": 49996894
        },
        {
          "time": "2026-08-28",
          "open": 213.79,
          "high": 215.76,
          "low": 212.9,
          "close": 214.87,
          "volume": 33370451
        },
        {
          "time": "2026-08-31",
          "open": 214.37,
          "high": 217.28,
          "low": 213.13,
          "close": 216.04,
          "volume": 20423023
        },
        {
          "time": "2026-09-01",
          "open": 215.54,
          "high": 219.08,
          "low": 214.06,
          "close": 217.6,
          "volume": 40385875
        },
        {
          "time": "2026-09-02",
          "open": 217.1,
          "high": 220.78,
          "low": 215.57,
          "close": 219.25,
          "volume": 48189096
        },
        {
          "time": "2026-09-03",
          "open": 218.75,
          "high": 222.05,
          "low": 217.36,
          "close": 220.66,
          "volume": 26215440
        },
        {
          "time": "2026-09-04",
          "open": 220.16,
          "high": 222.63,
          "low": 219.08,
          "close": 221.55,
          "volume": 24074588
        },
        {
          "time": "2026-09-07",
          "open": 221.05,
          "high": 222.46,
          "low": 220.37,
          "close": 221.77,
          "volume": 46599918
        },
        {
          "time": "2026-09-08",
          "open": 221.27,
          "high": 221.64,
          "low": 220.98,
          "close": 221.35,
          "volume": 42936240
        },
        {
          "time": "2026-09-09",
          "open": 220.85,
          "high": 221.32,
          "low": 219.99,
          "close": 220.46,
          "volume": 21355000
        },
        {
          "time": "2026-09-10",
          "open": 219.96,
          "high": 220.53,
          "low": 218.83,
          "close": 219.4,
          "volume": 30579926
        },
        {
          "time": "2026-09-11",
          "open": 218.9,
          "high": 219.37,
          "low": 218.05,
          "close": 218.52,
          "volume": 49784004
        },
        {
          "time": "2026-09-14",
          "open": 218.02,
          "high": 218.4,
          "low": 217.73,
          "close": 218.11,
          "volume": 35610399
        },
        {
          "time": "2026-09-15",
          "open": 217.61,
          "high": 219.03,
          "low": 216.93,
          "close": 218.35,
          "volume": 20058702
        },
        {
          "time": "2026-09-16",
          "open": 217.85,
          "high": 220.36,
          "low": 216.76,
          "close": 219.27,
          "volume": 38239809
        },
        {
          "time": "2026-09-17",
          "open": 218.77,
          "high": 222.15,
          "low": 217.35,
          "close": 220.73,
          "volume": 49106434
        },
        {
          "time": "2026-09-18",
          "open": 220.23,
          "high": 224.05,
          "low": 218.65,
          "close": 222.47,
          "volume": 28125118
        },
        {
          "time": "2026-09-21",
          "open": 221.97,
          "high": 225.7,
          "low": 220.42,
          "close": 224.15,
          "volume": 22665147
        },
        {
          "time": "2026-09-22",
          "open": 223.65,
          "high": 226.76,
          "low": 222.33,
          "close": 225.44,
          "volume": 45053438
        },
        {
          "time": "2026-09-23",
          "open": 224.94,
          "high": 227.05,
          "low": 224,
          "close": 226.11,
          "volume": 44744193
        },
        {
          "time": "2026-09-24",
          "open": 225.61,
          "high": 226.57,
          "low": 225.1,
          "close": 226.06,
          "volume": 22435591
        },
        {
          "time": "2026-09-25",
          "open": 225.55,
          "high": 225.89,
          "low": 225.05,
          "close": 225.39,
          "volume": 28493517
        },
        {
          "time": "2026-09-28",
          "open": 224.89,
          "high": 225.45,
          "low": 223.8,
          "close": 224.36,
          "volume": 49241058
        },
        {
          "time": "2026-09-29",
          "open": 223.86,
          "high": 224.43,
          "low": 222.74,
          "close": 223.31,
          "volume": 37836719
        },
        {
          "time": "2026-09-30",
          "open": 222.81,
          "high": 223.18,
          "low": 222.22,
          "close": 222.59,
          "volume": 20027951
        },
        {
          "time": "2026-10-01",
          "open": 222.09,
          "high": 222.92,
          "low": 221.63,
          "close": 222.46,
          "volume": 36021413
        },
        {
          "time": "2026-10-02",
          "open": 221.96,
          "high": 223.94,
          "low": 221.07,
          "close": 223.05,
          "volume": 49708842
        },
        {
          "time": "2026-10-05",
          "open": 222.55,
          "high": 225.59,
          "low": 221.26,
          "close": 224.3,
          "volume": 30188280
        }
      ],
      "ema20": [
        {
          "time": "2026-08-11",
          "value": 211.1
        },
        {
          "time": "2026-08-12",
          "value": 211.18
        },
        {
          "time": "2026-08-13",
          "value": 211.39
        },
        {
          "time": "2026-08-14",
          "value": 211.72
        },
        {
          "time": "2026-08-17",
          "value": 212.16
        },
        {
          "time": "2026-08-18",
          "value": 212.66
        },
        {
          "time": "2026-08-19",
          "value": 213.15
        },
        {
          "time": "2026-08-20",
          "value": 213.59
        },
        {
          "time": "2026-08-21",
          "value": 213.91
        },
        {
          "time": "2026-08-24",
          "value": 214.11
        },
        {
          "time": "2026-08-25",
          "value": 214.19
        },
        {
          "time": "2026-08-26",
          "value": 214.21
        },
        {
          "time": "2026-08-27",
          "value": 214.22
        },
        {
          "time": "2026-08-28",
          "value": 214.28
        },
        {
          "time": "2026-08-31",
          "value": 214.45
        },
        {
          "time": "2026-09-01",
          "value": 214.75
        },
        {
          "time": "2026-09-02",
          "value": 215.18
        },
        {
          "time": "2026-09-03",
          "value": 215.7
        },
        {
          "time": "2026-09-04",
          "value": 216.26
        },
        {
          "time": "2026-09-07",
          "value": 216.78
        },
        {
          "time": "2026-09-08",
          "value": 217.22
        },
        {
          "time": "2026-09-09",
          "value": 217.53
        },
        {
          "time": "2026-09-10",
          "value": 217.7
        },
        {
          "time": "2026-09-11",
          "value": 217.78
        },
        {
          "time": "2026-09-14",
          "value": 217.81
        },
        {
          "time": "2026-09-15",
          "value": 217.86
        },
        {
          "time": "2026-09-16",
          "value": 218
        },
        {
          "time": "2026-09-17",
          "value": 218.26
        },
        {
          "time": "2026-09-18",
          "value": 218.66
        },
        {
          "time": "2026-09-21",
          "value": 219.18
        },
        {
          "time": "2026-09-22",
          "value": 219.78
        },
        {
          "time": "2026-09-23",
          "value": 220.38
        },
        {
          "time": "2026-09-24",
          "value": 220.92
        },
        {
          "time": "2026-09-25",
          "value": 221.35
        },
        {
          "time": "2026-09-28",
          "value": 221.63
        },
        {
          "time": "2026-09-29",
          "value": 221.79
        },
        {
          "time": "2026-09-30",
          "value": 221.87
        },
        {
          "time": "2026-10-01",
          "value": 221.93
        },
        {
          "time": "2026-10-02",
          "value": 222.03
        },
        {
          "time": "2026-10-05",
          "value": 222.25
        }
      ],
      "chartMarkers": [
        {
          "time": "2026-09-18",
          "position": "belowBar",
          "color": "#3b82f6",
          "shape": "circle",
          "text": "FORM 4: T. Cook Share Grant"
        }
      ],
      "corporateEvent": {
        "coverage_status": "CONFIRMED_UPCOMING",
        "preflight_state": "PASS",
        "freshness_status": "LIVE",
        "days_to_earnings": 18.0,
        "has_imminent_earnings": false
      },
      "preflight": {
        "isApplicable": true,
        "isCleared": true,
        "passedCount": 5,
        "totalMandatory": 5,
        "activeBlocker": "NONE",
        "whatNeedsToChange": "Execution criteria satisfied. Simulated trade staging unlocked.",
        "evidenceFreshness": "LIVE",
        "evidenceCompleteness": "COMPLETE",
        "checks": [
          {
            "id": "CHK-1",
            "name": "Reward/Risk Asymmetry",
            "rule": "Reward:Risk >= 2.0:1",
            "status": "PASS",
            "badge": "PASS",
            "value": "2.8:1",
            "authority": "Canonical Decision Hierarchy (Trade Sizing Engine)",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Target 1 to Invalidation floor provides favorable 2.8:1 asymmetric convex payoff.",
            "recoveryAction": "Awaiting price retest within optimal accumulation buy zone."
          },
          {
            "id": "CHK-2",
            "name": "Technical Structure",
            "rule": "Stage 2 Base Breakout (Not Stage 4)",
            "status": "PASS",
            "badge": "PASS",
            "value": "Stage 2 Base (Confirmed)",
            "authority": "Minervini Volatility Contraction Pattern Analyzer",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Base contraction sequence complete; volume dried to institutional absorption levels.",
            "recoveryAction": "Wait for primary base consolidation to establish higher low floor."
          },
          {
            "id": "CHK-3",
            "name": "Capital Risk",
            "rule": "Short Float < 12% & Quality Score >= 60",
            "status": "PASS",
            "badge": "PASS",
            "value": "Short Float: 3.1% | Quality: 84/100",
            "authority": "Master Asset Catalog & Fundamental Solvency Ledger",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Low short interest and strong solvency metrics confirm absence of structural capital distress.",
            "recoveryAction": "Resolve incomplete balance sheet or short interest reporting."
          },
          {
            "id": "CHK-4",
            "name": "Catalyst Hazard",
            "rule": ">7 Days to Forward Binary Event",
            "status": "PASS",
            "badge": "PASS",
            "value": "18.0 Days to Binary Earnings (Confirmed)",
            "authority": "SEC EDGAR Form 8-K / Corporate Event Schedule",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Clean operational runway: Next corporate earnings report is 18.0 days away.",
            "recoveryAction": "Wait for binary catalyst risk event to pass (>7 days buffer required)."
          },
          {
            "id": "CHK-5",
            "name": "Macro Regime",
            "rule": "CBOE VIX Index < 26.0",
            "status": "PASS",
            "badge": "PASS",
            "value": "VIX 18.2",
            "authority": "CBOE Volatility Index & Canonical Macro Ribbon",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Macro volatility regime calm (VIX 18.2 < 26.0 threshold). Market conditions supportive.",
            "recoveryAction": "Wait for market-wide volatility index (VIX) to cool below 26.0 risk gate."
          }
        ]
      }
    },
    "SPY": {
      "state_id": "STATE_ETF_SPY_BENCHMARK",
      "symbol": "SPY",
      "name": "SPDR S&P 500 ETF Trust",
      "exchange": "NYSE Arca",
      "sector": "Index Benchmark",
      "secType": "ETF",
      "security_type": "ETF",
      "executionEligibility": "ETF_EXECUTION",
      "display_mode": "ETF_CONTEXT",
      "volumeStatement": "62.4M (102% vs 50D avg - Illustrative)",
      "currentPrice": 574.2,
      "priceDeltaDollar": 2.85,
      "priceDeltaPct": 0.5,
      "decisionState": "CORE_BENCHMARK_REGIME",
      "executionStatus": "ACCUMULATE_ON_DIPS",
      "isActionable": false,
      "pattern": "NOT_APPLICABLE_TO_ETF_CONTEXT",
      "confluenceScore": null,
      "confluenceRating": "CORE BENCHMARK CONTEXT",
      "confluenceBadge": "ETF Benchmark Context",
      "etfProfile": {
        "category": "US Large-Cap Core Equity Benchmark",
        "aum": "$560B",
        "expenseRatio": "0.09%",
        "trackingError": "0.04%",
        "beta": "1.00",
        "holdingsCount": "503",
        "topHoldingsConcentration": "32.4% (Top 10)",
        "dividendYield": "1.24%",
        "rebalanceFrequency": "Quarterly"
      },
      "macroContext": {
        "posture": "Bullish Secular Trend \u2014 Core Index Anchor",
        "regime": "Regime 1: Low Volatility Expansion",
        "trendState": "Bullish Secular Trend (> 50D / 200D SMA)",
        "benchmarkRole": "US Broad Equity Market Baseline",
        "yieldCurveSkew": "Normalizing (+14 bps 2s10s)",
        "sectorBreadth": "78% S&P Components > 50D SMA"
      },
      "levels": {
        "pivot": null,
        "entryMin": null,
        "entryMax": null,
        "referencePrice": 574.2,
        "stopLoss": null,
        "stopDistanceDollar": 0,
        "stopDistancePct": 0,
        "target1": null,
        "rewardDistanceDollar": 0,
        "rewardDistancePct": 0,
        "rewardRiskRatio": null,
        "riskRewardDisplay": "N/A (Benchmark Holding)",
        "note": "Tactical pivot levels and swing stops not applicable to benchmark holdings."
      },
      "sizing": {
        "shares": 0,
        "quantityAuthority": "NOT_APPLICABLE_TO_ETF_CONTEXT",
        "bindingConstraint": "Asset Allocation Model governs ETF benchmark weighting (Single-stock 2% swing risk sizer disabled).",
        "disclaimer": "ETF benchmark exposure is modeled under strategic asset allocation rather than tactical swing trading."
      },
      "pillars": [
        {
          "pillar": "MACRO_REGIME",
          "label": "Macro Economic Regime",
          "weightPct": 35,
          "score": 85,
          "summary": "Low Volatility Expansion",
          "state": "POSITIVE"
        },
        {
          "pillar": "MARKET_BREADTH",
          "label": "Index Market Breadth",
          "weightPct": 30,
          "score": 78,
          "summary": "78% Components > 50D SMA",
          "state": "POSITIVE"
        },
        {
          "pillar": "LIQUIDITY_FLOW",
          "label": "Institutional Liquidity",
          "weightPct": 20,
          "score": 82,
          "summary": "Consistent Primary Inflows",
          "state": "POSITIVE"
        },
        {
          "pillar": "VOLATILITY_SKEW",
          "label": "VIX Term Structure",
          "weightPct": 15,
          "score": 80,
          "summary": "Contango (VIX 14.8, 3M 16.2)",
          "state": "POSITIVE"
        }
      ],
      "smartMoney": {
        "headline": "Congressional Purchases Logged (House / Senate Financial Disclosures)",
        "trades": [
          {
            "date": "2026-09-18",
            "filer": "Rep. Nancy Pelosi (Spouse)",
            "type": "CONGRESS",
            "transaction": "BUY",
            "size": "$1,000,000 - $5,000,000",
            "disclosure": "STOCK Act Filing"
          },
          {
            "date": "2026-09-04",
            "filer": "Sen. Tommy Tuberville",
            "type": "CONGRESS",
            "transaction": "BUY",
            "size": "$250,000 - $500,000",
            "disclosure": "STOCK Act Filing"
          }
        ],
        "institutionalNotes": "Institutional rebalancing inflows steady; active dark pool blocks clustered around $570 support."
      },
      "solvency": {
        "isEtf": true,
        "etfRiskCard": {
          "expenseDrag": "0.09% annual fee ($9/year per $10k invested)",
          "turnoverRate": "2.0% annual turnover (minimal taxable drag)",
          "liquidityTier": "Ultra-High (Penny spreads, $60B+ daily volume)"
        }
      },
      "provenance": {
        "asOf": "2026-10-06 (Static Baseline Fixture)",
        "authority": "ARX Ephemeral Prototype Fixture Authority (Offline Simulation)",
        "quoteFreshness": "STATIC_OFFLINE_PROTOTYPE",
        "dataMode": "ILLUSTRATIVE_SYNTHETIC_FIXTURE",
        "marketDataFreshness": "NOT_APPLICABLE_TO_STATIC_PROTOTYPE"
      },
      "candles": [
        {
          "time": "2026-08-11",
          "open": 539.58,
          "high": 541.25,
          "low": 538.88,
          "close": 540.55,
          "volume": 35000000
        },
        {
          "time": "2026-08-12",
          "open": 540.37,
          "high": 543.29,
          "low": 539.2,
          "close": 542.12,
          "volume": 49874972
        },
        {
          "time": "2026-08-13",
          "open": 541.95,
          "high": 545.83,
          "low": 540.42,
          "close": 544.3,
          "volume": 31166883
        },
        {
          "time": "2026-08-14",
          "open": 544.12,
          "high": 548.42,
          "low": 542.43,
          "close": 546.73,
          "volume": 21112780
        },
        {
          "time": "2026-08-17",
          "open": 546.56,
          "high": 550.65,
          "low": 544.95,
          "close": 549.04,
          "volume": 42411700
        },
        {
          "time": "2026-08-18",
          "open": 548.86,
          "high": 552.15,
          "low": 547.55,
          "close": 550.84,
          "volume": 46977307
        },
        {
          "time": "2026-08-19",
          "open": 550.67,
          "high": 552.76,
          "low": 549.81,
          "close": 551.9,
          "volume": 24501880
        },
        {
          "time": "2026-08-20",
          "open": 551.72,
          "high": 552.5,
          "low": 551.35,
          "close": 552.13,
          "volume": 25727943
        },
        {
          "time": "2026-08-21",
          "open": 551.96,
          "high": 552.24,
          "low": 551.42,
          "close": 551.7,
          "volume": 47887427
        },
        {
          "time": "2026-08-24",
          "open": 551.53,
          "high": 552.03,
          "low": 550.4,
          "close": 550.9,
          "volume": 40951109
        },
        {
          "time": "2026-08-25",
          "open": 550.72,
          "high": 551.2,
          "low": 549.64,
          "close": 550.12,
          "volume": 20579038
        },
        {
          "time": "2026-08-26",
          "open": 549.95,
          "high": 550.17,
          "low": 549.56,
          "close": 549.78,
          "volume": 32765015
        },
        {
          "time": "2026-08-27",
          "open": 549.6,
          "high": 550.57,
          "low": 549.16,
          "close": 550.13,
          "volume": 49996894
        },
        {
          "time": "2026-08-28",
          "open": 549.96,
          "high": 552.24,
          "low": 549.03,
          "close": 551.31,
          "volume": 33370451
        },
        {
          "time": "2026-08-31",
          "open": 551.13,
          "high": 554.59,
          "low": 549.76,
          "close": 553.22,
          "volume": 20423023
        },
        {
          "time": "2026-09-01",
          "open": 553.05,
          "high": 557.27,
          "low": 551.39,
          "close": 555.61,
          "volume": 40385875
        },
        {
          "time": "2026-09-02",
          "open": 555.43,
          "high": 559.81,
          "low": 553.71,
          "close": 558.09,
          "volume": 48189096
        },
        {
          "time": "2026-09-03",
          "open": 557.92,
          "high": 561.8,
          "low": 556.39,
          "close": 560.27,
          "volume": 26215440
        },
        {
          "time": "2026-09-04",
          "open": 560.09,
          "high": 562.95,
          "low": 558.94,
          "close": 561.8,
          "volume": 24074588
        },
        {
          "time": "2026-09-07",
          "open": 561.63,
          "high": 563.17,
          "low": 560.98,
          "close": 562.52,
          "volume": 46599918
        },
        {
          "time": "2026-09-08",
          "open": 562.35,
          "high": 562.63,
          "low": 562.17,
          "close": 562.45,
          "volume": 42936240
        },
        {
          "time": "2026-09-09",
          "open": 562.27,
          "high": 562.68,
          "low": 561.38,
          "close": 561.79,
          "volume": 21355000
        },
        {
          "time": "2026-09-10",
          "open": 561.62,
          "high": 562.15,
          "low": 560.41,
          "close": 560.94,
          "volume": 30579926
        },
        {
          "time": "2026-09-11",
          "open": 560.76,
          "high": 561.16,
          "low": 559.89,
          "close": 560.29,
          "volume": 49784004
        },
        {
          "time": "2026-09-14",
          "open": 560.12,
          "high": 560.41,
          "low": 559.94,
          "close": 560.23,
          "volume": 35610399
        },
        {
          "time": "2026-09-15",
          "open": 560.05,
          "high": 561.61,
          "low": 559.39,
          "close": 560.95,
          "volume": 20058702
        },
        {
          "time": "2026-09-16",
          "open": 560.78,
          "high": 563.67,
          "low": 559.62,
          "close": 562.51,
          "volume": 38239809
        },
        {
          "time": "2026-09-17",
          "open": 562.33,
          "high": 566.27,
          "low": 560.78,
          "close": 564.72,
          "volume": 49106434
        },
        {
          "time": "2026-09-18",
          "open": 564.55,
          "high": 569.02,
          "low": 562.8,
          "close": 567.27,
          "volume": 28125118
        },
        {
          "time": "2026-09-21",
          "open": 567.1,
          "high": 571.44,
          "low": 565.4,
          "close": 569.74,
          "volume": 22665147
        },
        {
          "time": "2026-09-22",
          "open": 569.56,
          "high": 573.13,
          "low": 568.15,
          "close": 571.72,
          "volume": 45053438
        },
        {
          "time": "2026-09-23",
          "open": 571.55,
          "high": 573.91,
          "low": 570.59,
          "close": 572.95,
          "volume": 44744193
        },
        {
          "time": "2026-09-24",
          "open": 572.77,
          "high": 573.76,
          "low": 572.32,
          "close": 573.31,
          "volume": 22435591
        },
        {
          "time": "2026-09-25",
          "open": 573.14,
          "high": 573.38,
          "low": 572.71,
          "close": 572.95,
          "volume": 28493517
        },
        {
          "time": "2026-09-28",
          "open": 572.77,
          "high": 573.27,
          "low": 571.64,
          "close": 572.14,
          "volume": 49241058
        },
        {
          "time": "2026-09-29",
          "open": 571.97,
          "high": 572.48,
          "low": 570.81,
          "close": 571.32,
          "volume": 37836719
        },
        {
          "time": "2026-09-30",
          "open": 571.14,
          "high": 571.42,
          "low": 570.6,
          "close": 570.88,
          "volume": 20027951
        },
        {
          "time": "2026-10-01",
          "open": 570.71,
          "high": 571.55,
          "low": 570.32,
          "close": 571.16,
          "volume": 36021413
        },
        {
          "time": "2026-10-02",
          "open": 570.98,
          "high": 573.18,
          "low": 570.08,
          "close": 572.28,
          "volume": 49708842
        },
        {
          "time": "2026-10-05",
          "open": 572.11,
          "high": 575.58,
          "low": 570.73,
          "close": 574.2,
          "volume": 30188280
        }
      ],
      "chartMarkers": [
        {
          "time": "2026-09-18",
          "position": "aboveBar",
          "color": "#f59e0b",
          "shape": "arrowUp",
          "text": "CONGRESS: Rep. Pelosi BUY ($1M-$5M)"
        },
        {
          "time": "2026-09-04",
          "position": "aboveBar",
          "color": "#f59e0b",
          "shape": "arrowUp",
          "text": "CONGRESS: Sen. Tuberville BUY ($250k-$500k)"
        }
      ],
      "ema20": [
        {
          "time": "2026-08-11",
          "value": 540.55
        },
        {
          "time": "2026-08-12",
          "value": 540.7
        },
        {
          "time": "2026-08-13",
          "value": 541.04
        },
        {
          "time": "2026-08-14",
          "value": 541.58
        },
        {
          "time": "2026-08-17",
          "value": 542.29
        },
        {
          "time": "2026-08-18",
          "value": 543.11
        },
        {
          "time": "2026-08-19",
          "value": 543.95
        },
        {
          "time": "2026-08-20",
          "value": 544.72
        },
        {
          "time": "2026-08-21",
          "value": 545.39
        },
        {
          "time": "2026-08-24",
          "value": 545.91
        },
        {
          "time": "2026-08-25",
          "value": 546.31
        },
        {
          "time": "2026-08-26",
          "value": 546.64
        },
        {
          "time": "2026-08-27",
          "value": 546.98
        },
        {
          "time": "2026-08-28",
          "value": 547.39
        },
        {
          "time": "2026-08-31",
          "value": 547.94
        },
        {
          "time": "2026-09-01",
          "value": 548.67
        },
        {
          "time": "2026-09-02",
          "value": 549.57
        },
        {
          "time": "2026-09-03",
          "value": 550.59
        },
        {
          "time": "2026-09-04",
          "value": 551.66
        },
        {
          "time": "2026-09-07",
          "value": 552.69
        },
        {
          "time": "2026-09-08",
          "value": 553.62
        },
        {
          "time": "2026-09-09",
          "value": 554.4
        },
        {
          "time": "2026-09-10",
          "value": 555.02
        },
        {
          "time": "2026-09-11",
          "value": 555.52
        },
        {
          "time": "2026-09-14",
          "value": 555.97
        },
        {
          "time": "2026-09-15",
          "value": 556.45
        },
        {
          "time": "2026-09-16",
          "value": 557.02
        },
        {
          "time": "2026-09-17",
          "value": 557.76
        },
        {
          "time": "2026-09-18",
          "value": 558.66
        },
        {
          "time": "2026-09-21",
          "value": 559.72
        },
        {
          "time": "2026-09-22",
          "value": 560.86
        },
        {
          "time": "2026-09-23",
          "value": 562.01
        },
        {
          "time": "2026-09-24",
          "value": 563.09
        },
        {
          "time": "2026-09-25",
          "value": 564.03
        },
        {
          "time": "2026-09-28",
          "value": 564.8
        },
        {
          "time": "2026-09-29",
          "value": 565.42
        },
        {
          "time": "2026-09-30",
          "value": 565.94
        },
        {
          "time": "2026-10-01",
          "value": 566.44
        },
        {
          "time": "2026-10-02",
          "value": 566.99
        },
        {
          "time": "2026-10-05",
          "value": 567.68
        }
      ],
      "preflight": {
        "isApplicable": false,
        "isCleared": false,
        "passedCount": 0,
        "totalMandatory": 0,
        "activeBlocker": "Benchmark ETF Non-Applicability (Index Instrument)",
        "whatNeedsToChange": "Benchmark ETFs track broad market baskets; single-stock equity Pre-Flight rules do not apply.",
        "evidenceFreshness": "LIVE",
        "evidenceCompleteness": "COMPLETE",
        "checks": [
          {
            "id": "CHK-1",
            "name": "Reward/Risk Asymmetry",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Single-stock reward-to-risk rules do not apply to diversified index baskets.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-2",
            "name": "Technical Structure",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Single-stock base pattern rules do not apply to benchmark indexes.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-3",
            "name": "Capital Risk",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Short squeeze and balance sheet insolvency rules do not apply to index funds.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-4",
            "name": "Catalyst Hazard",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Index baskets do not have single-stock discrete binary earnings releases.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-5",
            "name": "Macro Regime",
            "rule": "Macro Context",
            "status": "PASS",
            "badge": "PASS",
            "value": "VIX 18.2",
            "authority": "CBOE Volatility Index",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Macro reference benchmark tracking broader market regime.",
            "recoveryAction": "None (Benchmark Role)."
          }
        ]
      }
    },
    "QQQ": {
      "state_id": "STATE_ETF_QQQ_BENCHMARK",
      "symbol": "QQQ",
      "name": "Invesco QQQ Trust",
      "exchange": "NASDAQ",
      "sector": "Index Benchmark",
      "secType": "ETF",
      "security_type": "ETF",
      "executionEligibility": "ETF_EXECUTION",
      "display_mode": "ETF_CONTEXT",
      "volumeStatement": "38.2M (115% vs 50D avg - Illustrative)",
      "currentPrice": 488.6,
      "priceDeltaDollar": 4.1,
      "priceDeltaPct": 0.85,
      "decisionState": "CORE_BENCHMARK_REGIME",
      "executionStatus": "ACCUMULATE_ON_DIPS",
      "isActionable": false,
      "pattern": "NOT_APPLICABLE_TO_ETF_CONTEXT",
      "confluenceScore": null,
      "confluenceRating": "CORE BENCHMARK CONTEXT",
      "confluenceBadge": "ETF Benchmark Context",
      "etfProfile": {
        "category": "Nasdaq-100 Large-Cap Growth Benchmark",
        "aum": "$285B",
        "expenseRatio": "0.20%",
        "trackingError": "0.06%",
        "beta": "1.18",
        "holdingsCount": "101",
        "topHoldingsConcentration": "51.2% (Top 10)",
        "dividendYield": "0.58%",
        "rebalanceFrequency": "Quarterly"
      },
      "macroContext": {
        "posture": "Bullish Tech Expansion \u2014 Secular Growth Anchor",
        "regime": "Regime 1: Secular Tech Capex Wave",
        "trendState": "Bullish Uptrend (> 20D / 50D SMA)",
        "benchmarkRole": "Large-Cap Growth & Innovation Baseline",
        "yieldCurveSkew": "Technology Duration Resilient",
        "sectorBreadth": "82% Nasdaq-100 Components > 50D SMA"
      },
      "levels": {
        "pivot": null,
        "entryMin": null,
        "entryMax": null,
        "referencePrice": 488.6,
        "stopLoss": null,
        "stopDistanceDollar": 0,
        "stopDistancePct": 0,
        "target1": null,
        "rewardDistanceDollar": 0,
        "rewardDistancePct": 0,
        "rewardRiskRatio": null,
        "riskRewardDisplay": "N/A (Benchmark Holding)",
        "note": "Tactical pivot levels and swing stops not applicable to benchmark holdings."
      },
      "sizing": {
        "shares": 0,
        "quantityAuthority": "NOT_APPLICABLE_TO_ETF_CONTEXT",
        "bindingConstraint": "Asset Allocation Model governs ETF benchmark weighting (Single-stock 2% swing risk sizer disabled).",
        "disclaimer": "ETF benchmark exposure is modeled under strategic asset allocation rather than tactical swing trading."
      },
      "pillars": [
        {
          "pillar": "MACRO_REGIME",
          "label": "Macro Economic Regime",
          "weightPct": 35,
          "score": 88,
          "summary": "Secular Tech Capex Wave",
          "state": "POSITIVE"
        },
        {
          "pillar": "MARKET_BREADTH",
          "label": "Tech Sector Breadth",
          "weightPct": 30,
          "score": 82,
          "summary": "82% Components > 50D SMA",
          "state": "POSITIVE"
        },
        {
          "pillar": "LIQUIDITY_FLOW",
          "label": "Institutional Inflows",
          "weightPct": 20,
          "score": 85,
          "summary": "AI Infrastructure Capital Commitments",
          "state": "POSITIVE"
        },
        {
          "pillar": "VOLATILITY_SKEW",
          "label": "VXN Skew",
          "weightPct": 15,
          "score": 82,
          "summary": "Low Put Skew / Call Open Interest Leading",
          "state": "POSITIVE"
        }
      ],
      "smartMoney": {
        "headline": "Congressional Purchases Logged (House / Senate Financial Disclosures)",
        "trades": [
          {
            "date": "2026-09-22",
            "filer": "Rep. Nancy Pelosi (Spouse)",
            "type": "CONGRESS",
            "transaction": "BUY",
            "size": "$500,000 - $1,000,000",
            "disclosure": "STOCK Act Filing"
          }
        ],
        "institutionalNotes": "Aggressive mutual fund and family office allocation to QQQ following enterprise AI margin expansion."
      },
      "solvency": {
        "isEtf": true,
        "etfRiskCard": {
          "expenseDrag": "0.20% annual fee ($20/year per $10k invested)",
          "turnoverRate": "8.0% annual turnover",
          "liquidityTier": "Ultra-High (Penny spreads, $20B+ daily volume)"
        }
      },
      "provenance": {
        "asOf": "2026-10-06 (Static Baseline Fixture)",
        "authority": "ARX Ephemeral Prototype Fixture Authority (Offline Simulation)",
        "quoteFreshness": "STATIC_OFFLINE_PROTOTYPE",
        "dataMode": "ILLUSTRATIVE_SYNTHETIC_FIXTURE",
        "marketDataFreshness": "NOT_APPLICABLE_TO_STATIC_PROTOTYPE"
      },
      "candles": [
        {
          "time": "2026-08-11",
          "open": 448.71,
          "high": 451.48,
          "low": 447.56,
          "close": 450.33,
          "volume": 35000000
        },
        {
          "time": "2026-08-12",
          "open": 449.54,
          "high": 453.88,
          "low": 447.79,
          "close": 452.14,
          "volume": 49874972
        },
        {
          "time": "2026-08-13",
          "open": 451.34,
          "high": 456.88,
          "low": 449.15,
          "close": 454.69,
          "volume": 31166883
        },
        {
          "time": "2026-08-14",
          "open": 453.89,
          "high": 459.99,
          "low": 451.49,
          "close": 457.59,
          "volume": 21112780
        },
        {
          "time": "2026-08-17",
          "open": 456.79,
          "high": 462.64,
          "low": 454.48,
          "close": 460.33,
          "volume": 42411700
        },
        {
          "time": "2026-08-18",
          "open": 459.54,
          "high": 464.39,
          "low": 457.61,
          "close": 462.46,
          "volume": 46977307
        },
        {
          "time": "2026-08-19",
          "open": 461.66,
          "high": 465.01,
          "low": 460.29,
          "close": 463.64,
          "volume": 24501880
        },
        {
          "time": "2026-08-20",
          "open": 462.84,
          "high": 464.54,
          "low": 462.09,
          "close": 463.79,
          "volume": 25727943
        },
        {
          "time": "2026-08-21",
          "open": 462.99,
          "high": 463.31,
          "low": 462.76,
          "close": 463.08,
          "volume": 47887427
        },
        {
          "time": "2026-08-24",
          "open": 462.29,
          "high": 462.7,
          "low": 461.5,
          "close": 461.91,
          "volume": 40951109
        },
        {
          "time": "2026-08-25",
          "open": 461.11,
          "high": 461.49,
          "low": 460.39,
          "close": 460.77,
          "volume": 20579038
        },
        {
          "time": "2026-08-26",
          "open": 459.97,
          "high": 460.47,
          "low": 459.67,
          "close": 460.17,
          "volume": 32765015
        },
        {
          "time": "2026-08-27",
          "open": 459.37,
          "high": 461.31,
          "low": 458.53,
          "close": 460.47,
          "volume": 49996894
        },
        {
          "time": "2026-08-28",
          "open": 459.67,
          "high": 463.29,
          "low": 458.21,
          "close": 461.83,
          "volume": 33370451
        },
        {
          "time": "2026-08-31",
          "open": 461.03,
          "high": 466.17,
          "low": 458.99,
          "close": 464.13,
          "volume": 20423023
        },
        {
          "time": "2026-09-01",
          "open": 463.33,
          "high": 469.46,
          "low": 460.92,
          "close": 467.05,
          "volume": 40385875
        },
        {
          "time": "2026-09-02",
          "open": 466.25,
          "high": 472.61,
          "low": 463.75,
          "close": 470.11,
          "volume": 48189096
        },
        {
          "time": "2026-09-03",
          "open": 469.32,
          "high": 475.07,
          "low": 467.04,
          "close": 472.79,
          "volume": 26215440
        },
        {
          "time": "2026-09-04",
          "open": 472,
          "high": 476.43,
          "low": 470.23,
          "close": 474.66,
          "volume": 24074588
        },
        {
          "time": "2026-09-07",
          "open": 473.86,
          "high": 476.6,
          "low": 472.72,
          "close": 475.46,
          "volume": 46599918
        },
        {
          "time": "2026-09-08",
          "open": 474.66,
          "high": 475.75,
          "low": 474.14,
          "close": 475.23,
          "volume": 42936240
        },
        {
          "time": "2026-09-09",
          "open": 474.44,
          "high": 474.73,
          "low": 473.97,
          "close": 474.26,
          "volume": 21355000
        },
        {
          "time": "2026-09-10",
          "open": 473.46,
          "high": 473.91,
          "low": 472.57,
          "close": 473.02,
          "volume": 30579926
        },
        {
          "time": "2026-09-11",
          "open": 472.22,
          "high": 472.5,
          "low": 471.77,
          "close": 472.05,
          "volume": 49784004
        },
        {
          "time": "2026-09-14",
          "open": 471.25,
          "high": 472.36,
          "low": 470.72,
          "close": 471.83,
          "volume": 35610399
        },
        {
          "time": "2026-09-15",
          "open": 471.04,
          "high": 473.81,
          "low": 469.89,
          "close": 472.66,
          "volume": 20058702
        },
        {
          "time": "2026-09-16",
          "open": 471.86,
          "high": 476.38,
          "low": 470.05,
          "close": 474.57,
          "volume": 38239809
        },
        {
          "time": "2026-09-17",
          "open": 473.77,
          "high": 479.68,
          "low": 471.44,
          "close": 477.35,
          "volume": 49106434
        },
        {
          "time": "2026-09-18",
          "open": 476.55,
          "high": 483.18,
          "low": 473.95,
          "close": 480.58,
          "volume": 28125118
        },
        {
          "time": "2026-09-21",
          "open": 479.79,
          "high": 486.26,
          "low": 477.25,
          "close": 483.72,
          "volume": 22665147
        },
        {
          "time": "2026-09-22",
          "open": 482.92,
          "high": 488.4,
          "low": 480.75,
          "close": 486.23,
          "volume": 45053438
        },
        {
          "time": "2026-09-23",
          "open": 485.43,
          "high": 489.31,
          "low": 483.86,
          "close": 487.74,
          "volume": 44744193
        },
        {
          "time": "2026-09-24",
          "open": 486.94,
          "high": 489.01,
          "low": 486.05,
          "close": 488.12,
          "volume": 22435591
        },
        {
          "time": "2026-09-25",
          "open": 487.33,
          "high": 487.83,
          "low": 487.03,
          "close": 487.53,
          "volume": 28493517
        },
        {
          "time": "2026-09-28",
          "open": 486.73,
          "high": 487.13,
          "low": 485.96,
          "close": 486.36,
          "volume": 49241058
        },
        {
          "time": "2026-09-29",
          "open": 485.56,
          "high": 485.98,
          "low": 484.73,
          "close": 485.15,
          "volume": 37836719
        },
        {
          "time": "2026-09-30",
          "open": 484.35,
          "high": 484.71,
          "low": 484.1,
          "close": 484.46,
          "volume": 20027951
        },
        {
          "time": "2026-10-01",
          "open": 483.67,
          "high": 485.54,
          "low": 482.86,
          "close": 484.73,
          "volume": 36021413
        },
        {
          "time": "2026-10-02",
          "open": 483.93,
          "high": 487.63,
          "low": 482.43,
          "close": 486.13,
          "volume": 49708842
        },
        {
          "time": "2026-10-05",
          "open": 485.33,
          "high": 490.74,
          "low": 483.19,
          "close": 488.6,
          "volume": 30188280
        }
      ],
      "chartMarkers": [
        {
          "time": "2026-09-22",
          "position": "aboveBar",
          "color": "#f59e0b",
          "shape": "arrowUp",
          "text": "CONGRESS: Rep. Pelosi BUY ($500k-$1M)"
        }
      ],
      "ema20": [
        {
          "time": "2026-08-11",
          "value": 450.33
        },
        {
          "time": "2026-08-12",
          "value": 450.5
        },
        {
          "time": "2026-08-13",
          "value": 450.9
        },
        {
          "time": "2026-08-14",
          "value": 451.54
        },
        {
          "time": "2026-08-17",
          "value": 452.38
        },
        {
          "time": "2026-08-18",
          "value": 453.34
        },
        {
          "time": "2026-08-19",
          "value": 454.32
        },
        {
          "time": "2026-08-20",
          "value": 455.22
        },
        {
          "time": "2026-08-21",
          "value": 455.97
        },
        {
          "time": "2026-08-24",
          "value": 456.53
        },
        {
          "time": "2026-08-25",
          "value": 456.94
        },
        {
          "time": "2026-08-26",
          "value": 457.25
        },
        {
          "time": "2026-08-27",
          "value": 457.55
        },
        {
          "time": "2026-08-28",
          "value": 457.96
        },
        {
          "time": "2026-08-31",
          "value": 458.55
        },
        {
          "time": "2026-09-01",
          "value": 459.36
        },
        {
          "time": "2026-09-02",
          "value": 460.38
        },
        {
          "time": "2026-09-03",
          "value": 461.56
        },
        {
          "time": "2026-09-04",
          "value": 462.81
        },
        {
          "time": "2026-09-07",
          "value": 464.02
        },
        {
          "time": "2026-09-08",
          "value": 465.08
        },
        {
          "time": "2026-09-09",
          "value": 465.96
        },
        {
          "time": "2026-09-10",
          "value": 466.63
        },
        {
          "time": "2026-09-11",
          "value": 467.15
        },
        {
          "time": "2026-09-14",
          "value": 467.59
        },
        {
          "time": "2026-09-15",
          "value": 468.07
        },
        {
          "time": "2026-09-16",
          "value": 468.69
        },
        {
          "time": "2026-09-17",
          "value": 469.52
        },
        {
          "time": "2026-09-18",
          "value": 470.57
        },
        {
          "time": "2026-09-21",
          "value": 471.82
        },
        {
          "time": "2026-09-22",
          "value": 473.2
        },
        {
          "time": "2026-09-23",
          "value": 474.58
        },
        {
          "time": "2026-09-24",
          "value": 475.87
        },
        {
          "time": "2026-09-25",
          "value": 476.98
        },
        {
          "time": "2026-09-28",
          "value": 477.87
        },
        {
          "time": "2026-09-29",
          "value": 478.57
        },
        {
          "time": "2026-09-30",
          "value": 479.13
        },
        {
          "time": "2026-10-01",
          "value": 479.66
        },
        {
          "time": "2026-10-02",
          "value": 480.28
        },
        {
          "time": "2026-10-05",
          "value": 481.07
        }
      ],
      "preflight": {
        "isApplicable": false,
        "isCleared": false,
        "passedCount": 0,
        "totalMandatory": 0,
        "activeBlocker": "Benchmark ETF Non-Applicability (Index Instrument)",
        "whatNeedsToChange": "Benchmark ETFs track broad market baskets; single-stock equity Pre-Flight rules do not apply.",
        "evidenceFreshness": "LIVE",
        "evidenceCompleteness": "COMPLETE",
        "checks": [
          {
            "id": "CHK-1",
            "name": "Reward/Risk Asymmetry",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Single-stock reward-to-risk rules do not apply to diversified index baskets.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-2",
            "name": "Technical Structure",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Single-stock base pattern rules do not apply to benchmark indexes.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-3",
            "name": "Capital Risk",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Short squeeze and balance sheet insolvency rules do not apply to index funds.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-4",
            "name": "Catalyst Hazard",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Index baskets do not have single-stock discrete binary earnings releases.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-5",
            "name": "Macro Regime",
            "rule": "Macro Context",
            "status": "PASS",
            "badge": "PASS",
            "value": "VIX 18.2",
            "authority": "CBOE Volatility Index",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Macro reference benchmark tracking broader market regime.",
            "recoveryAction": "None (Benchmark Role)."
          }
        ]
      }
    },
    "SMH": {
      "state_id": "STATE_ETF_SMH_BENCHMARK",
      "symbol": "SMH",
      "name": "VanEck Semiconductor ETF",
      "exchange": "NASDAQ",
      "sector": "Semiconductors",
      "secType": "ETF",
      "security_type": "ETF",
      "executionEligibility": "ETF_EXECUTION",
      "display_mode": "ETF_CONTEXT",
      "volumeStatement": "14.8M (122% vs 50D avg - Illustrative)",
      "currentPrice": 264.8,
      "priceDeltaDollar": 5.6,
      "priceDeltaPct": 2.16,
      "decisionState": "CORE_BENCHMARK_REGIME",
      "executionStatus": "ACCUMULATE_ON_DIPS",
      "isActionable": false,
      "pattern": "NOT_APPLICABLE_TO_ETF_CONTEXT",
      "confluenceScore": null,
      "confluenceRating": "SECTOR BENCHMARK CONTEXT",
      "confluenceBadge": "ETF Sector Context",
      "etfProfile": {
        "category": "Global Semiconductor Industry Benchmark",
        "aum": "$22B",
        "expenseRatio": "0.35%",
        "trackingError": "0.12%",
        "beta": "1.55",
        "holdingsCount": "26",
        "topHoldingsConcentration": "71.5% (Top 10)",
        "dividendYield": "0.45%",
        "rebalanceFrequency": "Semi-Annual"
      },
      "macroContext": {
        "posture": "Cyclical Tech Outperformance \u2014 Semiconductor Lead",
        "regime": "Regime 1: Capex Acceleration",
        "trendState": "Strong Uptrend (> 20D / 50D SMA)",
        "benchmarkRole": "Semiconductor Beta Satellite",
        "yieldCurveSkew": "High Beta Growth Exposure",
        "sectorBreadth": "88% Semi Components > 50D SMA"
      },
      "levels": {
        "pivot": null,
        "entryMin": null,
        "entryMax": null,
        "referencePrice": 264.8,
        "stopLoss": null,
        "stopDistanceDollar": 0,
        "stopDistancePct": 0,
        "target1": null,
        "rewardDistanceDollar": 0,
        "rewardDistancePct": 0,
        "rewardRiskRatio": null,
        "riskRewardDisplay": "N/A (Sector ETF)",
        "note": "Tactical pivot levels and swing stops not applicable to sector benchmark baskets."
      },
      "sizing": {
        "shares": 0,
        "quantityAuthority": "NOT_APPLICABLE_TO_ETF_CONTEXT",
        "bindingConstraint": "Sector Allocation Model governs ETF weighting (Single-stock 2% swing risk sizer disabled).",
        "disclaimer": "ETF sector exposure is modeled under strategic asset allocation rather than tactical swing trading."
      },
      "pillars": [
        {
          "pillar": "MACRO_REGIME",
          "label": "Semiconductor Cycle",
          "weightPct": 35,
          "score": 92,
          "summary": "Leading Datacenter Capex",
          "state": "POSITIVE"
        },
        {
          "pillar": "MARKET_BREADTH",
          "label": "Sector Breadth",
          "weightPct": 30,
          "score": 88,
          "summary": "88% Components > 50D SMA",
          "state": "POSITIVE"
        },
        {
          "pillar": "LIQUIDITY_FLOW",
          "label": "Institutional Flow",
          "weightPct": 20,
          "score": 90,
          "summary": "Heavy ETF Inflow Aggregation",
          "state": "POSITIVE"
        },
        {
          "pillar": "VOLATILITY_SKEW",
          "label": "Implied Volatility",
          "weightPct": 15,
          "score": 76,
          "summary": "Normal Volatility Skew",
          "state": "POSITIVE"
        }
      ],
      "smartMoney": {
        "headline": "Congressional Purchases Logged (House / Senate Financial Disclosures)",
        "trades": [],
        "institutionalNotes": "Global asset managers aggregating SMH as synthetic AI infrastructure proxy."
      },
      "solvency": {
        "isEtf": true,
        "etfRiskCard": {
          "expenseDrag": "0.35% annual fee ($35/year per $10k invested)",
          "turnoverRate": "12.0% annual turnover",
          "liquidityTier": "High ($2B+ daily volume)"
        }
      },
      "provenance": {
        "asOf": "2026-10-06 (Static Baseline Fixture)",
        "authority": "ARX Ephemeral Prototype Fixture Authority (Offline Simulation)",
        "quoteFreshness": "STATIC_OFFLINE_PROTOTYPE",
        "dataMode": "ILLUSTRATIVE_SYNTHETIC_FIXTURE",
        "marketDataFreshness": "NOT_APPLICABLE_TO_STATIC_PROTOTYPE"
      },
      "candles": [
        {
          "time": "2026-08-11",
          "open": 236.37,
          "high": 240.16,
          "low": 234.72,
          "close": 238.51,
          "volume": 35000000
        },
        {
          "time": "2026-08-12",
          "open": 236.56,
          "high": 242.04,
          "low": 234.28,
          "close": 239.76,
          "volume": 49874972
        },
        {
          "time": "2026-08-13",
          "open": 237.8,
          "high": 244.62,
          "low": 235.02,
          "close": 241.84,
          "volume": 31166883
        },
        {
          "time": "2026-08-14",
          "open": 239.89,
          "high": 247.37,
          "low": 236.86,
          "close": 244.34,
          "volume": 21112780
        },
        {
          "time": "2026-08-17",
          "open": 242.39,
          "high": 249.68,
          "low": 239.43,
          "close": 246.72,
          "volume": 42411700
        },
        {
          "time": "2026-08-18",
          "open": 244.77,
          "high": 251.03,
          "low": 242.2,
          "close": 248.46,
          "volume": 46977307
        },
        {
          "time": "2026-08-19",
          "open": 246.5,
          "high": 251.13,
          "low": 244.54,
          "close": 249.17,
          "volume": 24501880
        },
        {
          "time": "2026-08-20",
          "open": 247.22,
          "high": 250.04,
          "low": 245.94,
          "close": 248.76,
          "volume": 25727943
        },
        {
          "time": "2026-08-21",
          "open": 246.81,
          "high": 248.1,
          "low": 246.1,
          "close": 247.39,
          "volume": 47887427
        },
        {
          "time": "2026-08-24",
          "open": 245.44,
          "high": 245.88,
          "low": 245.05,
          "close": 245.49,
          "volume": 40951109
        },
        {
          "time": "2026-08-25",
          "open": 243.54,
          "high": 244.06,
          "low": 243.12,
          "close": 243.64,
          "volume": 20579038
        },
        {
          "time": "2026-08-26",
          "open": 241.68,
          "high": 243.16,
          "low": 240.9,
          "close": 242.38,
          "volume": 32765015
        },
        {
          "time": "2026-08-27",
          "open": 240.43,
          "high": 243.54,
          "low": 239.04,
          "close": 242.15,
          "volume": 49996894
        },
        {
          "time": "2026-08-28",
          "open": 240.2,
          "high": 245.25,
          "low": 238.08,
          "close": 243.13,
          "volume": 33370451
        },
        {
          "time": "2026-08-31",
          "open": 241.18,
          "high": 247.99,
          "low": 238.4,
          "close": 245.21,
          "volume": 20423023
        },
        {
          "time": "2026-09-01",
          "open": 243.25,
          "high": 251.27,
          "low": 240.02,
          "close": 248.04,
          "volume": 40385875
        },
        {
          "time": "2026-09-02",
          "open": 246.09,
          "high": 254.45,
          "low": 242.73,
          "close": 251.09,
          "volume": 48189096
        },
        {
          "time": "2026-09-03",
          "open": 249.14,
          "high": 256.86,
          "low": 246.02,
          "close": 253.74,
          "volume": 26215440
        },
        {
          "time": "2026-09-04",
          "open": 251.79,
          "high": 258.02,
          "low": 249.23,
          "close": 255.46,
          "volume": 24074588
        },
        {
          "time": "2026-09-07",
          "open": 253.51,
          "high": 257.77,
          "low": 251.69,
          "close": 255.95,
          "volume": 46599918
        },
        {
          "time": "2026-09-08",
          "open": 253.99,
          "high": 256.29,
          "low": 252.9,
          "close": 255.2,
          "volume": 42936240
        },
        {
          "time": "2026-09-09",
          "open": 253.25,
          "high": 254.12,
          "low": 252.7,
          "close": 253.57,
          "volume": 21355000
        },
        {
          "time": "2026-09-10",
          "open": 251.62,
          "high": 251.98,
          "low": 251.26,
          "close": 251.62,
          "volume": 30579926
        },
        {
          "time": "2026-09-11",
          "open": 249.67,
          "high": 250.56,
          "low": 249.11,
          "close": 250,
          "volume": 49784004
        },
        {
          "time": "2026-09-14",
          "open": 248.04,
          "high": 250.38,
          "low": 246.94,
          "close": 249.28,
          "volume": 35610399
        },
        {
          "time": "2026-09-15",
          "open": 247.33,
          "high": 251.69,
          "low": 245.47,
          "close": 249.83,
          "volume": 20058702
        },
        {
          "time": "2026-09-16",
          "open": 247.88,
          "high": 254.39,
          "low": 245.21,
          "close": 251.72,
          "volume": 38239809
        },
        {
          "time": "2026-09-17",
          "open": 249.77,
          "high": 258.04,
          "low": 246.44,
          "close": 254.71,
          "volume": 49106434
        },
        {
          "time": "2026-09-18",
          "open": 252.75,
          "high": 261.98,
          "low": 249.06,
          "close": 258.29,
          "volume": 28125118
        },
        {
          "time": "2026-09-21",
          "open": 256.34,
          "high": 265.47,
          "low": 252.69,
          "close": 261.82,
          "volume": 22665147
        },
        {
          "time": "2026-09-22",
          "open": 259.87,
          "high": 267.85,
          "low": 256.65,
          "close": 264.63,
          "volume": 45053438
        },
        {
          "time": "2026-09-23",
          "open": 262.68,
          "high": 268.69,
          "low": 260.2,
          "close": 266.21,
          "volume": 44744193
        },
        {
          "time": "2026-09-24",
          "open": 264.26,
          "high": 267.99,
          "low": 262.64,
          "close": 266.37,
          "volume": 22435591
        },
        {
          "time": "2026-09-25",
          "open": 264.41,
          "high": 266.14,
          "low": 263.54,
          "close": 265.27,
          "volume": 28493517
        },
        {
          "time": "2026-09-28",
          "open": 263.32,
          "high": 263.87,
          "low": 262.89,
          "close": 263.44,
          "volume": 49241058
        },
        {
          "time": "2026-09-29",
          "open": 261.49,
          "high": 261.96,
          "low": 261.09,
          "close": 261.56,
          "volume": 37836719
        },
        {
          "time": "2026-09-30",
          "open": 259.61,
          "high": 261.16,
          "low": 258.8,
          "close": 260.35,
          "volume": 20027951
        },
        {
          "time": "2026-10-01",
          "open": 258.39,
          "high": 261.9,
          "low": 256.85,
          "close": 260.36,
          "volume": 36021413
        },
        {
          "time": "2026-10-02",
          "open": 258.41,
          "high": 264.31,
          "low": 255.97,
          "close": 261.87,
          "volume": 49708842
        },
        {
          "time": "2026-10-05",
          "open": 259.92,
          "high": 268.09,
          "low": 256.63,
          "close": 264.8,
          "volume": 30188280
        }
      ],
      "chartMarkers": [],
      "ema20": [
        {
          "time": "2026-08-11",
          "value": 238.51
        },
        {
          "time": "2026-08-12",
          "value": 238.63
        },
        {
          "time": "2026-08-13",
          "value": 238.93
        },
        {
          "time": "2026-08-14",
          "value": 239.45
        },
        {
          "time": "2026-08-17",
          "value": 240.14
        },
        {
          "time": "2026-08-18",
          "value": 240.93
        },
        {
          "time": "2026-08-19",
          "value": 241.72
        },
        {
          "time": "2026-08-20",
          "value": 242.39
        },
        {
          "time": "2026-08-21",
          "value": 242.87
        },
        {
          "time": "2026-08-24",
          "value": 243.12
        },
        {
          "time": "2026-08-25",
          "value": 243.17
        },
        {
          "time": "2026-08-26",
          "value": 243.09
        },
        {
          "time": "2026-08-27",
          "value": 243
        },
        {
          "time": "2026-08-28",
          "value": 243.01
        },
        {
          "time": "2026-08-31",
          "value": 243.22
        },
        {
          "time": "2026-09-01",
          "value": 243.68
        },
        {
          "time": "2026-09-02",
          "value": 244.39
        },
        {
          "time": "2026-09-03",
          "value": 245.28
        },
        {
          "time": "2026-09-04",
          "value": 246.25
        },
        {
          "time": "2026-09-07",
          "value": 247.17
        },
        {
          "time": "2026-09-08",
          "value": 247.94
        },
        {
          "time": "2026-09-09",
          "value": 248.47
        },
        {
          "time": "2026-09-10",
          "value": 248.77
        },
        {
          "time": "2026-09-11",
          "value": 248.89
        },
        {
          "time": "2026-09-14",
          "value": 248.93
        },
        {
          "time": "2026-09-15",
          "value": 249.01
        },
        {
          "time": "2026-09-16",
          "value": 249.27
        },
        {
          "time": "2026-09-17",
          "value": 249.79
        },
        {
          "time": "2026-09-18",
          "value": 250.6
        },
        {
          "time": "2026-09-21",
          "value": 251.67
        },
        {
          "time": "2026-09-22",
          "value": 252.9
        },
        {
          "time": "2026-09-23",
          "value": 254.17
        },
        {
          "time": "2026-09-24",
          "value": 255.33
        },
        {
          "time": "2026-09-25",
          "value": 256.28
        },
        {
          "time": "2026-09-28",
          "value": 256.96
        },
        {
          "time": "2026-09-29",
          "value": 257.4
        },
        {
          "time": "2026-09-30",
          "value": 257.68
        },
        {
          "time": "2026-10-01",
          "value": 257.93
        },
        {
          "time": "2026-10-02",
          "value": 258.31
        },
        {
          "time": "2026-10-05",
          "value": 258.93
        }
      ],
      "preflight": {
        "isApplicable": false,
        "isCleared": false,
        "passedCount": 0,
        "totalMandatory": 0,
        "activeBlocker": "Benchmark ETF Non-Applicability (Index Instrument)",
        "whatNeedsToChange": "Benchmark ETFs track broad market baskets; single-stock equity Pre-Flight rules do not apply.",
        "evidenceFreshness": "LIVE",
        "evidenceCompleteness": "COMPLETE",
        "checks": [
          {
            "id": "CHK-1",
            "name": "Reward/Risk Asymmetry",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Single-stock reward-to-risk rules do not apply to diversified index baskets.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-2",
            "name": "Technical Structure",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Single-stock base pattern rules do not apply to benchmark indexes.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-3",
            "name": "Capital Risk",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Short squeeze and balance sheet insolvency rules do not apply to index funds.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-4",
            "name": "Catalyst Hazard",
            "rule": "Single-Stock N/A",
            "status": "NA",
            "badge": "N/A",
            "value": "N/A (ETF Basket)",
            "authority": "ARX Benchmark Authority",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Index baskets do not have single-stock discrete binary earnings releases.",
            "recoveryAction": "None (Benchmark Role)."
          },
          {
            "id": "CHK-5",
            "name": "Macro Regime",
            "rule": "Macro Context",
            "status": "PASS",
            "badge": "PASS",
            "value": "VIX 18.2",
            "authority": "CBOE Volatility Index",
            "timestamp": "2026-10-24 10:00 EDT",
            "explanation": "Macro reference benchmark tracking broader market regime.",
            "recoveryAction": "None (Benchmark Role)."
          }
        ]
      }
    },
    "TSLA": {
      "state_id": "STATE_5_ALREADY_HELD_MONITORING",
      "symbol": "TSLA",
      "name": "Tesla, Inc.",
      "exchange": "NASDAQ",
      "sector": "Automotive & Clean Energy Tech",
      "volumeStatement": "68.2M (115% vs 50D avg)",
      "currentPrice": 245.5,
      "priceDeltaDollar": 5.2,
      "priceDeltaPct": 2.16,
      "decisionState": "ACTIONABLE_SETUP",
      "executionStatus": "IN_BUY_ZONE",
      "isActionable": true,
      "newEntryActionable": false,
      "uiPresentationMode": "POSITION_MONITORING",
      "positionMonitoringCanonical": false,
      "pattern": "Cup & Handle Consolidation",
      "confluenceScore": 78,
      "confluenceRating": "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT",
      "confluenceBadge": "4-Pillar Confluence (Pristine)",
      "levels": {
        "pivot": 242.0,
        "entryMin": 240.0,
        "entryMax": 246.0,
        "referencePrice": 245.5,
        "stopLoss": 234.0,
        "stopDistanceDollar": 11.5,
        "stopDistancePct": -4.68,
        "target1": 273.1,
        "rewardDistanceDollar": 27.6,
        "rewardDistancePct": 11.24,
        "rewardRiskRatio": 2.4,
        "riskRewardDisplay": "1 : 2.40",
        "rMultipleTarget1": 2.4
      },
      "sizing": {
        "shares": 80,
        "capitalAllocated": 19640.0,
        "portfolioAllocationPct": 15.78,
        "modeledStopLossDollar": 920.0,
        "modeledStopLossEquityPct": 0.74,
        "riskBudgetUtilizationPct": 36.95
      },
      "pillars": [
        {
          "pillar": "TECHNICAL_STRUCTURE",
          "label": "Technical Confluence",
          "weightPct": 25,
          "score": 86,
          "status": "positive",
          "detail": "Price advancing along 10 EMA; approaching Target 1 ($228.00)."
        },
        {
          "pillar": "FUNDAMENTAL_SOLVENCY",
          "label": "Fundamental Moat",
          "weightPct": 25,
          "score": 92,
          "status": "positive",
          "detail": "ROIC >50%; massive free cash flow generation; active share buyback."
        },
        {
          "pillar": "SMART_MONEY_FLOW",
          "label": "Institutional Flow",
          "weightPct": 25,
          "score": 80,
          "status": "positive",
          "detail": "Consistent institutional index and ETF rebalancing inflows."
        },
        {
          "pillar": "MACRO_SAFETY_FLOOR",
          "label": "Macro Environment",
          "weightPct": 25,
          "score": 78,
          "status": "positive",
          "detail": "Tech sector beta supportive."
        }
      ],
      "provenance": {
        "asOf": "2026-10-06 (Static Baseline Fixture)",
        "authority": "ARX Ephemeral Prototype Fixture Authority (Offline Simulation)",
        "quoteFreshness": "STATIC_OFFLINE_PROTOTYPE",
        "dataMode": "ILLUSTRATIVE_SYNTHETIC_FIXTURE",
        "marketDataFreshness": "NOT_APPLICABLE_TO_STATIC_PROTOTYPE"
      },
      "secType": "COMMON_STOCK",
      "security_type": "COMMON_STOCK",
      "display_mode": "SINGLE_STOCK_ANALYSIS",
      "executionEligibility": "STOCK_EXECUTION",
      "candles": [
        {
          "time": "2026-08-11",
          "open": 210.34,
          "high": 211.8,
          "low": 209.64,
          "close": 211.1,
          "volume": 35000000
        },
        {
          "time": "2026-08-12",
          "open": 210.6,
          "high": 213.03,
          "low": 209.54,
          "close": 211.97,
          "volume": 49874972
        },
        {
          "time": "2026-08-13",
          "open": 211.47,
          "high": 214.66,
          "low": 210.12,
          "close": 213.31,
          "volume": 31166883
        },
        {
          "time": "2026-08-14",
          "open": 212.81,
          "high": 216.35,
          "low": 211.33,
          "close": 214.87,
          "volume": 21112780
        },
        {
          "time": "2026-08-17",
          "open": 214.37,
          "high": 217.75,
          "low": 212.95,
          "close": 216.33,
          "volume": 42411700
        },
        {
          "time": "2026-08-18",
          "open": 215.83,
          "high": 218.58,
          "low": 214.65,
          "close": 217.4,
          "volume": 46977307
        },
        {
          "time": "2026-08-19",
          "open": 216.89,
          "high": 218.7,
          "low": 216.06,
          "close": 217.87,
          "volume": 24501880
        },
        {
          "time": "2026-08-20",
          "open": 217.37,
          "high": 218.14,
          "low": 216.93,
          "close": 217.7,
          "volume": 25727943
        },
        {
          "time": "2026-08-21",
          "open": 217.2,
          "high": 217.57,
          "low": 216.62,
          "close": 216.99,
          "volume": 47887427
        },
        {
          "time": "2026-08-24",
          "open": 216.49,
          "high": 217.03,
          "low": 215.45,
          "close": 215.99,
          "volume": 40951109
        },
        {
          "time": "2026-08-25",
          "open": 215.49,
          "high": 216.02,
          "low": 214.48,
          "close": 215.01,
          "volume": 20579038
        },
        {
          "time": "2026-08-26",
          "open": 214.51,
          "high": 214.83,
          "low": 214.05,
          "close": 214.37,
          "volume": 32765015
        },
        {
          "time": "2026-08-27",
          "open": 213.87,
          "high": 214.78,
          "low": 213.38,
          "close": 214.29,
          "volume": 49996894
        },
        {
          "time": "2026-08-28",
          "open": 213.79,
          "high": 215.76,
          "low": 212.9,
          "close": 214.87,
          "volume": 33370451
        },
        {
          "time": "2026-08-31",
          "open": 214.37,
          "high": 217.28,
          "low": 213.13,
          "close": 216.04,
          "volume": 20423023
        },
        {
          "time": "2026-09-01",
          "open": 215.54,
          "high": 219.08,
          "low": 214.06,
          "close": 217.6,
          "volume": 40385875
        },
        {
          "time": "2026-09-02",
          "open": 217.1,
          "high": 220.78,
          "low": 215.57,
          "close": 219.25,
          "volume": 48189096
        },
        {
          "time": "2026-09-03",
          "open": 218.75,
          "high": 222.05,
          "low": 217.36,
          "close": 220.66,
          "volume": 26215440
        },
        {
          "time": "2026-09-04",
          "open": 220.16,
          "high": 222.63,
          "low": 219.08,
          "close": 221.55,
          "volume": 24074588
        },
        {
          "time": "2026-09-07",
          "open": 221.05,
          "high": 222.46,
          "low": 220.37,
          "close": 221.77,
          "volume": 46599918
        },
        {
          "time": "2026-09-08",
          "open": 221.27,
          "high": 221.64,
          "low": 220.98,
          "close": 221.35,
          "volume": 42936240
        },
        {
          "time": "2026-09-09",
          "open": 220.85,
          "high": 221.32,
          "low": 219.99,
          "close": 220.46,
          "volume": 21355000
        },
        {
          "time": "2026-09-10",
          "open": 219.96,
          "high": 220.53,
          "low": 218.83,
          "close": 219.4,
          "volume": 30579926
        },
        {
          "time": "2026-09-11",
          "open": 218.9,
          "high": 219.37,
          "low": 218.05,
          "close": 218.52,
          "volume": 49784004
        },
        {
          "time": "2026-09-14",
          "open": 218.02,
          "high": 218.4,
          "low": 217.73,
          "close": 218.11,
          "volume": 35610399
        },
        {
          "time": "2026-09-15",
          "open": 217.61,
          "high": 219.03,
          "low": 216.93,
          "close": 218.35,
          "volume": 20058702
        },
        {
          "time": "2026-09-16",
          "open": 217.85,
          "high": 220.36,
          "low": 216.76,
          "close": 219.27,
          "volume": 38239809
        },
        {
          "time": "2026-09-17",
          "open": 218.77,
          "high": 222.15,
          "low": 217.35,
          "close": 220.73,
          "volume": 49106434
        },
        {
          "time": "2026-09-18",
          "open": 220.23,
          "high": 224.05,
          "low": 218.65,
          "close": 222.47,
          "volume": 28125118
        },
        {
          "time": "2026-09-21",
          "open": 221.97,
          "high": 225.7,
          "low": 220.42,
          "close": 224.15,
          "volume": 22665147
        },
        {
          "time": "2026-09-22",
          "open": 223.65,
          "high": 226.76,
          "low": 222.33,
          "close": 225.44,
          "volume": 45053438
        },
        {
          "time": "2026-09-23",
          "open": 224.94,
          "high": 227.05,
          "low": 224,
          "close": 226.11,
          "volume": 44744193
        },
        {
          "time": "2026-09-24",
          "open": 225.61,
          "high": 226.57,
          "low": 225.1,
          "close": 226.06,
          "volume": 22435591
        },
        {
          "time": "2026-09-25",
          "open": 225.55,
          "high": 225.89,
          "low": 225.05,
          "close": 225.39,
          "volume": 28493517
        },
        {
          "time": "2026-09-28",
          "open": 224.89,
          "high": 225.45,
          "low": 223.8,
          "close": 224.36,
          "volume": 49241058
        },
        {
          "time": "2026-09-29",
          "open": 223.86,
          "high": 224.43,
          "low": 222.74,
          "close": 223.31,
          "volume": 37836719
        },
        {
          "time": "2026-09-30",
          "open": 222.81,
          "high": 223.18,
          "low": 222.22,
          "close": 222.59,
          "volume": 20027951
        },
        {
          "time": "2026-10-01",
          "open": 222.09,
          "high": 222.92,
          "low": 221.63,
          "close": 222.46,
          "volume": 36021413
        },
        {
          "time": "2026-10-02",
          "open": 221.96,
          "high": 223.94,
          "low": 221.07,
          "close": 223.05,
          "volume": 49708842
        },
        {
          "time": "2026-10-05",
          "open": 222.55,
          "high": 225.59,
          "low": 221.26,
          "close": 224.3,
          "volume": 30188280
        }
      ],
      "ema20": [
        {
          "time": "2026-08-11",
          "value": 211.1
        },
        {
          "time": "2026-08-12",
          "value": 211.18
        },
        {
          "time": "2026-08-13",
          "value": 211.39
        },
        {
          "time": "2026-08-14",
          "value": 211.72
        },
        {
          "time": "2026-08-17",
          "value": 212.16
        },
        {
          "time": "2026-08-18",
          "value": 212.66
        },
        {
          "time": "2026-08-19",
          "value": 213.15
        },
        {
          "time": "2026-08-20",
          "value": 213.59
        },
        {
          "time": "2026-08-21",
          "value": 213.91
        },
        {
          "time": "2026-08-24",
          "value": 214.11
        },
        {
          "time": "2026-08-25",
          "value": 214.19
        },
        {
          "time": "2026-08-26",
          "value": 214.21
        },
        {
          "time": "2026-08-27",
          "value": 214.22
        },
        {
          "time": "2026-08-28",
          "value": 214.28
        },
        {
          "time": "2026-08-31",
          "value": 214.45
        },
        {
          "time": "2026-09-01",
          "value": 214.75
        },
        {
          "time": "2026-09-02",
          "value": 215.18
        },
        {
          "time": "2026-09-03",
          "value": 215.7
        },
        {
          "time": "2026-09-04",
          "value": 216.26
        },
        {
          "time": "2026-09-07",
          "value": 216.78
        },
        {
          "time": "2026-09-08",
          "value": 217.22
        },
        {
          "time": "2026-09-09",
          "value": 217.53
        },
        {
          "time": "2026-09-10",
          "value": 217.7
        },
        {
          "time": "2026-09-11",
          "value": 217.78
        },
        {
          "time": "2026-09-14",
          "value": 217.81
        },
        {
          "time": "2026-09-15",
          "value": 217.86
        },
        {
          "time": "2026-09-16",
          "value": 218
        },
        {
          "time": "2026-09-17",
          "value": 218.26
        },
        {
          "time": "2026-09-18",
          "value": 218.66
        },
        {
          "time": "2026-09-21",
          "value": 219.18
        },
        {
          "time": "2026-09-22",
          "value": 219.78
        },
        {
          "time": "2026-09-23",
          "value": 220.38
        },
        {
          "time": "2026-09-24",
          "value": 220.92
        },
        {
          "time": "2026-09-25",
          "value": 221.35
        },
        {
          "time": "2026-09-28",
          "value": 221.63
        },
        {
          "time": "2026-09-29",
          "value": 221.79
        },
        {
          "time": "2026-09-30",
          "value": 221.87
        },
        {
          "time": "2026-10-01",
          "value": 221.93
        },
        {
          "time": "2026-10-02",
          "value": 222.03
        },
        {
          "time": "2026-10-05",
          "value": 222.25
        }
      ],
      "chartMarkers": [
        {
          "time": "2026-09-18",
          "position": "belowBar",
          "color": "#3b82f6",
          "shape": "circle",
          "text": "FORM 4: T. Cook Share Grant"
        }
      ],
      "corporateEvent": null,
      "preflight": {
        "isApplicable": true,
        "isCleared": false,
        "passedCount": 4,
        "totalMandatory": 5,
        "activeBlocker": "Catalyst Coverage Unavailable (Forward Earnings Unverified)",
        "whatNeedsToChange": "Await verified forward corporate event and earnings calendar coverage.",
        "evidenceFreshness": "UNAVAILABLE",
        "evidenceCompleteness": "PARTIAL",
        "checks": [
          {
            "id": "CHK-1",
            "name": "Reward/Risk Asymmetry",
            "rule": "Reward:Risk >= 2.0:1",
            "status": "PASS",
            "badge": "PASS",
            "value": "2.4:1",
            "authority": "Canonical Decision Hierarchy (Trade Sizing Engine)",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Target 1 to Invalidation satisfies 2.4:1 asymmetric ratio.",
            "recoveryAction": "Awaiting price retest within optimal accumulation buy zone."
          },
          {
            "id": "CHK-2",
            "name": "Technical Structure",
            "rule": "Stage 2 Base Breakout (Not Stage 4)",
            "status": "PASS",
            "badge": "PASS",
            "value": "Cup & Handle (Confirmed)",
            "authority": "Minervini Volatility Contraction Pattern Analyzer",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Handle breakout confirmed above $242.00 pivot.",
            "recoveryAction": "Wait for primary base consolidation to establish higher low floor."
          },
          {
            "id": "CHK-3",
            "name": "Capital Risk",
            "rule": "Short Float < 12% & Quality Score >= 60",
            "status": "PASS",
            "badge": "PASS",
            "value": "Short Float: 4.2% | Quality: 78/100",
            "authority": "Master Asset Catalog & Fundamental Solvency Ledger",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Short interest 4.2% within acceptable limits.",
            "recoveryAction": "Resolve incomplete balance sheet or short interest reporting."
          },
          {
            "id": "CHK-4",
            "name": "Catalyst Hazard",
            "rule": ">7 Days to Forward Binary Event",
            "status": "UNAVAILABLE",
            "badge": "UNAVAILABLE",
            "value": "Coverage Unavailable",
            "authority": "SEC EDGAR Corporate Events Authority",
            "timestamp": "2026-10-24 10:00 EDT - UNAVAILABLE",
            "explanation": "Catalyst evidence unavailable: Forward earnings calendar unverified.",
            "recoveryAction": "Await verified forward corporate event and earnings calendar coverage."
          },
          {
            "id": "CHK-5",
            "name": "Macro Regime",
            "rule": "CBOE VIX Index < 26.0",
            "status": "PASS",
            "badge": "PASS",
            "value": "VIX 19.2",
            "authority": "CBOE Volatility Index & Canonical Macro Ribbon",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Macro regime calm (VIX 19.2 < 26.0).",
            "recoveryAction": "Wait for market-wide volatility index (VIX) to cool below 26.0 risk gate."
          }
        ]
      },
      "catalystForecast": null
    },
    "AMZN": {
      "state_id": "STATE_5_ALREADY_HELD_MONITORING",
      "symbol": "AMZN",
      "name": "Amazon.com, Inc.",
      "exchange": "NASDAQ",
      "sector": "E-Commerce & Cloud Infrastructure",
      "volumeStatement": "45.1M (104% vs 50D avg)",
      "currentPrice": 186.2,
      "priceDeltaDollar": 2.1,
      "priceDeltaPct": 1.14,
      "decisionState": "ACTIONABLE_SETUP",
      "executionStatus": "IN_BUY_ZONE",
      "isActionable": true,
      "newEntryActionable": false,
      "uiPresentationMode": "POSITION_MONITORING",
      "positionMonitoringCanonical": false,
      "pattern": "High-Tight Flag Breakout",
      "confluenceScore": 82,
      "confluenceRating": "HIGH-CONVICTION INSTITUTIONAL ALIGNMENT",
      "confluenceBadge": "4-Pillar Confluence (Pristine)",
      "levels": {
        "pivot": 184.0,
        "entryMin": 182.0,
        "entryMax": 187.0,
        "referencePrice": 186.2,
        "stopLoss": 178.5,
        "stopDistanceDollar": 7.7,
        "stopDistancePct": -4.14,
        "target1": 206.2,
        "rewardDistanceDollar": 20.0,
        "rewardDistancePct": 10.74,
        "rewardRiskRatio": 2.6,
        "riskRewardDisplay": "1 : 2.60",
        "rMultipleTarget1": 2.6
      },
      "sizing": {
        "shares": 100,
        "capitalAllocated": 18620.0,
        "portfolioAllocationPct": 14.96,
        "modeledStopLossDollar": 770.0,
        "modeledStopLossEquityPct": 0.62,
        "riskBudgetUtilizationPct": 30.92
      },
      "pillars": [
        {
          "pillar": "TECHNICAL_STRUCTURE",
          "label": "Technical Confluence",
          "weightPct": 25,
          "score": 86,
          "status": "positive",
          "detail": "Price advancing along 10 EMA; approaching Target 1 ($228.00)."
        },
        {
          "pillar": "FUNDAMENTAL_SOLVENCY",
          "label": "Fundamental Moat",
          "weightPct": 25,
          "score": 92,
          "status": "positive",
          "detail": "ROIC >50%; massive free cash flow generation; active share buyback."
        },
        {
          "pillar": "SMART_MONEY_FLOW",
          "label": "Institutional Flow",
          "weightPct": 25,
          "score": 80,
          "status": "positive",
          "detail": "Consistent institutional index and ETF rebalancing inflows."
        },
        {
          "pillar": "MACRO_SAFETY_FLOOR",
          "label": "Macro Environment",
          "weightPct": 25,
          "score": 78,
          "status": "positive",
          "detail": "Tech sector beta supportive."
        }
      ],
      "provenance": {
        "asOf": "2026-10-06 (Static Baseline Fixture)",
        "authority": "ARX Ephemeral Prototype Fixture Authority (Offline Simulation)",
        "quoteFreshness": "STATIC_OFFLINE_PROTOTYPE",
        "dataMode": "ILLUSTRATIVE_SYNTHETIC_FIXTURE",
        "marketDataFreshness": "NOT_APPLICABLE_TO_STATIC_PROTOTYPE"
      },
      "secType": "COMMON_STOCK",
      "security_type": "COMMON_STOCK",
      "display_mode": "SINGLE_STOCK_ANALYSIS",
      "executionEligibility": "STOCK_EXECUTION",
      "candles": [
        {
          "time": "2026-08-11",
          "open": 210.34,
          "high": 211.8,
          "low": 209.64,
          "close": 211.1,
          "volume": 35000000
        },
        {
          "time": "2026-08-12",
          "open": 210.6,
          "high": 213.03,
          "low": 209.54,
          "close": 211.97,
          "volume": 49874972
        },
        {
          "time": "2026-08-13",
          "open": 211.47,
          "high": 214.66,
          "low": 210.12,
          "close": 213.31,
          "volume": 31166883
        },
        {
          "time": "2026-08-14",
          "open": 212.81,
          "high": 216.35,
          "low": 211.33,
          "close": 214.87,
          "volume": 21112780
        },
        {
          "time": "2026-08-17",
          "open": 214.37,
          "high": 217.75,
          "low": 212.95,
          "close": 216.33,
          "volume": 42411700
        },
        {
          "time": "2026-08-18",
          "open": 215.83,
          "high": 218.58,
          "low": 214.65,
          "close": 217.4,
          "volume": 46977307
        },
        {
          "time": "2026-08-19",
          "open": 216.89,
          "high": 218.7,
          "low": 216.06,
          "close": 217.87,
          "volume": 24501880
        },
        {
          "time": "2026-08-20",
          "open": 217.37,
          "high": 218.14,
          "low": 216.93,
          "close": 217.7,
          "volume": 25727943
        },
        {
          "time": "2026-08-21",
          "open": 217.2,
          "high": 217.57,
          "low": 216.62,
          "close": 216.99,
          "volume": 47887427
        },
        {
          "time": "2026-08-24",
          "open": 216.49,
          "high": 217.03,
          "low": 215.45,
          "close": 215.99,
          "volume": 40951109
        },
        {
          "time": "2026-08-25",
          "open": 215.49,
          "high": 216.02,
          "low": 214.48,
          "close": 215.01,
          "volume": 20579038
        },
        {
          "time": "2026-08-26",
          "open": 214.51,
          "high": 214.83,
          "low": 214.05,
          "close": 214.37,
          "volume": 32765015
        },
        {
          "time": "2026-08-27",
          "open": 213.87,
          "high": 214.78,
          "low": 213.38,
          "close": 214.29,
          "volume": 49996894
        },
        {
          "time": "2026-08-28",
          "open": 213.79,
          "high": 215.76,
          "low": 212.9,
          "close": 214.87,
          "volume": 33370451
        },
        {
          "time": "2026-08-31",
          "open": 214.37,
          "high": 217.28,
          "low": 213.13,
          "close": 216.04,
          "volume": 20423023
        },
        {
          "time": "2026-09-01",
          "open": 215.54,
          "high": 219.08,
          "low": 214.06,
          "close": 217.6,
          "volume": 40385875
        },
        {
          "time": "2026-09-02",
          "open": 217.1,
          "high": 220.78,
          "low": 215.57,
          "close": 219.25,
          "volume": 48189096
        },
        {
          "time": "2026-09-03",
          "open": 218.75,
          "high": 222.05,
          "low": 217.36,
          "close": 220.66,
          "volume": 26215440
        },
        {
          "time": "2026-09-04",
          "open": 220.16,
          "high": 222.63,
          "low": 219.08,
          "close": 221.55,
          "volume": 24074588
        },
        {
          "time": "2026-09-07",
          "open": 221.05,
          "high": 222.46,
          "low": 220.37,
          "close": 221.77,
          "volume": 46599918
        },
        {
          "time": "2026-09-08",
          "open": 221.27,
          "high": 221.64,
          "low": 220.98,
          "close": 221.35,
          "volume": 42936240
        },
        {
          "time": "2026-09-09",
          "open": 220.85,
          "high": 221.32,
          "low": 219.99,
          "close": 220.46,
          "volume": 21355000
        },
        {
          "time": "2026-09-10",
          "open": 219.96,
          "high": 220.53,
          "low": 218.83,
          "close": 219.4,
          "volume": 30579926
        },
        {
          "time": "2026-09-11",
          "open": 218.9,
          "high": 219.37,
          "low": 218.05,
          "close": 218.52,
          "volume": 49784004
        },
        {
          "time": "2026-09-14",
          "open": 218.02,
          "high": 218.4,
          "low": 217.73,
          "close": 218.11,
          "volume": 35610399
        },
        {
          "time": "2026-09-15",
          "open": 217.61,
          "high": 219.03,
          "low": 216.93,
          "close": 218.35,
          "volume": 20058702
        },
        {
          "time": "2026-09-16",
          "open": 217.85,
          "high": 220.36,
          "low": 216.76,
          "close": 219.27,
          "volume": 38239809
        },
        {
          "time": "2026-09-17",
          "open": 218.77,
          "high": 222.15,
          "low": 217.35,
          "close": 220.73,
          "volume": 49106434
        },
        {
          "time": "2026-09-18",
          "open": 220.23,
          "high": 224.05,
          "low": 218.65,
          "close": 222.47,
          "volume": 28125118
        },
        {
          "time": "2026-09-21",
          "open": 221.97,
          "high": 225.7,
          "low": 220.42,
          "close": 224.15,
          "volume": 22665147
        },
        {
          "time": "2026-09-22",
          "open": 223.65,
          "high": 226.76,
          "low": 222.33,
          "close": 225.44,
          "volume": 45053438
        },
        {
          "time": "2026-09-23",
          "open": 224.94,
          "high": 227.05,
          "low": 224,
          "close": 226.11,
          "volume": 44744193
        },
        {
          "time": "2026-09-24",
          "open": 225.61,
          "high": 226.57,
          "low": 225.1,
          "close": 226.06,
          "volume": 22435591
        },
        {
          "time": "2026-09-25",
          "open": 225.55,
          "high": 225.89,
          "low": 225.05,
          "close": 225.39,
          "volume": 28493517
        },
        {
          "time": "2026-09-28",
          "open": 224.89,
          "high": 225.45,
          "low": 223.8,
          "close": 224.36,
          "volume": 49241058
        },
        {
          "time": "2026-09-29",
          "open": 223.86,
          "high": 224.43,
          "low": 222.74,
          "close": 223.31,
          "volume": 37836719
        },
        {
          "time": "2026-09-30",
          "open": 222.81,
          "high": 223.18,
          "low": 222.22,
          "close": 222.59,
          "volume": 20027951
        },
        {
          "time": "2026-10-01",
          "open": 222.09,
          "high": 222.92,
          "low": 221.63,
          "close": 222.46,
          "volume": 36021413
        },
        {
          "time": "2026-10-02",
          "open": 221.96,
          "high": 223.94,
          "low": 221.07,
          "close": 223.05,
          "volume": 49708842
        },
        {
          "time": "2026-10-05",
          "open": 222.55,
          "high": 225.59,
          "low": 221.26,
          "close": 224.3,
          "volume": 30188280
        }
      ],
      "ema20": [
        {
          "time": "2026-08-11",
          "value": 211.1
        },
        {
          "time": "2026-08-12",
          "value": 211.18
        },
        {
          "time": "2026-08-13",
          "value": 211.39
        },
        {
          "time": "2026-08-14",
          "value": 211.72
        },
        {
          "time": "2026-08-17",
          "value": 212.16
        },
        {
          "time": "2026-08-18",
          "value": 212.66
        },
        {
          "time": "2026-08-19",
          "value": 213.15
        },
        {
          "time": "2026-08-20",
          "value": 213.59
        },
        {
          "time": "2026-08-21",
          "value": 213.91
        },
        {
          "time": "2026-08-24",
          "value": 214.11
        },
        {
          "time": "2026-08-25",
          "value": 214.19
        },
        {
          "time": "2026-08-26",
          "value": 214.21
        },
        {
          "time": "2026-08-27",
          "value": 214.22
        },
        {
          "time": "2026-08-28",
          "value": 214.28
        },
        {
          "time": "2026-08-31",
          "value": 214.45
        },
        {
          "time": "2026-09-01",
          "value": 214.75
        },
        {
          "time": "2026-09-02",
          "value": 215.18
        },
        {
          "time": "2026-09-03",
          "value": 215.7
        },
        {
          "time": "2026-09-04",
          "value": 216.26
        },
        {
          "time": "2026-09-07",
          "value": 216.78
        },
        {
          "time": "2026-09-08",
          "value": 217.22
        },
        {
          "time": "2026-09-09",
          "value": 217.53
        },
        {
          "time": "2026-09-10",
          "value": 217.7
        },
        {
          "time": "2026-09-11",
          "value": 217.78
        },
        {
          "time": "2026-09-14",
          "value": 217.81
        },
        {
          "time": "2026-09-15",
          "value": 217.86
        },
        {
          "time": "2026-09-16",
          "value": 218
        },
        {
          "time": "2026-09-17",
          "value": 218.26
        },
        {
          "time": "2026-09-18",
          "value": 218.66
        },
        {
          "time": "2026-09-21",
          "value": 219.18
        },
        {
          "time": "2026-09-22",
          "value": 219.78
        },
        {
          "time": "2026-09-23",
          "value": 220.38
        },
        {
          "time": "2026-09-24",
          "value": 220.92
        },
        {
          "time": "2026-09-25",
          "value": 221.35
        },
        {
          "time": "2026-09-28",
          "value": 221.63
        },
        {
          "time": "2026-09-29",
          "value": 221.79
        },
        {
          "time": "2026-09-30",
          "value": 221.87
        },
        {
          "time": "2026-10-01",
          "value": 221.93
        },
        {
          "time": "2026-10-02",
          "value": 222.03
        },
        {
          "time": "2026-10-05",
          "value": 222.25
        }
      ],
      "chartMarkers": [
        {
          "time": "2026-09-18",
          "position": "belowBar",
          "color": "#3b82f6",
          "shape": "circle",
          "text": "FORM 4: T. Cook Share Grant"
        }
      ],
      "corporateEvent": {
        "coverage_status": "CONFIRMED_UPCOMING",
        "preflight_state": "PASS",
        "freshness_status": "LIVE",
        "days_to_earnings": 24.0,
        "has_imminent_earnings": false
      },
      "preflight": {
        "isApplicable": true,
        "isCleared": false,
        "passedCount": 4,
        "totalMandatory": 5,
        "activeBlocker": "Macro Volatility Hazard (CBOE VIX 28.5 >= 26.0)",
        "whatNeedsToChange": "Wait for macro volatility index (VIX) to cool below 26.0 threshold.",
        "evidenceFreshness": "LIVE",
        "evidenceCompleteness": "COMPLETE",
        "checks": [
          {
            "id": "CHK-1",
            "name": "Reward/Risk Asymmetry",
            "rule": "Reward:Risk >= 2.0:1",
            "status": "PASS",
            "badge": "PASS",
            "value": "2.6:1",
            "authority": "Canonical Decision Hierarchy (Trade Sizing Engine)",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Target 1 to Invalidation satisfies 2.6:1 asymmetric ratio.",
            "recoveryAction": "Awaiting price retest within optimal accumulation buy zone."
          },
          {
            "id": "CHK-2",
            "name": "Technical Structure",
            "rule": "Stage 2 Base Breakout (Not Stage 4)",
            "status": "PASS",
            "badge": "PASS",
            "value": "High-Tight Flag (Confirmed)",
            "authority": "Minervini Volatility Contraction Pattern Analyzer",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "High-tight flag breakout confirmed above $184.00 pivot.",
            "recoveryAction": "Wait for primary base consolidation to establish higher low floor."
          },
          {
            "id": "CHK-3",
            "name": "Capital Risk",
            "rule": "Short Float < 12% & Quality Score >= 60",
            "status": "PASS",
            "badge": "PASS",
            "value": "Short Float: 2.1% | Quality: 88/100",
            "authority": "Master Asset Catalog & Fundamental Solvency Ledger",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Low short float (2.1%) and high fundamental quality (88).",
            "recoveryAction": "Resolve incomplete balance sheet or short interest reporting."
          },
          {
            "id": "CHK-4",
            "name": "Catalyst Hazard",
            "rule": ">7 Days to Forward Binary Event",
            "status": "PASS",
            "badge": "PASS",
            "value": "24.0 Days to Binary Earnings (Confirmed)",
            "authority": "SEC EDGAR Corporate Events Authority",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "Clean forward earnings window: 24.0 days buffer.",
            "recoveryAction": "Wait for binary catalyst risk event to pass (>7 days buffer required)."
          },
          {
            "id": "CHK-5",
            "name": "Macro Regime",
            "rule": "CBOE VIX Index < 26.0",
            "status": "FAIL",
            "badge": "HIGH VIX",
            "value": "VIX 28.5 (Elevated)",
            "authority": "CBOE Volatility Index & Canonical Macro Ribbon",
            "timestamp": "2026-10-24 10:00 EDT - LIVE",
            "explanation": "\u26a0\ufe0f Macro Volatility Hazard: CBOE VIX 28.5 exceeds maximum risk threshold (26.0).",
            "recoveryAction": "Wait for macro volatility index (VIX) to cool below 26.0 threshold."
          }
        ]
      }
    }
  }
};

if (typeof window !== 'undefined') {
  window.ARX_SYNTHESIS_FIXTURES = ARX_SYNTHESIS_FIXTURES;
}
