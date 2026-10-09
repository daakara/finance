"""ARX VCP Domain Numeric Contract.

Sprint 2B Domain-Authority Resolution.
Freezes deterministic numeric semantics, precision, calculations, and rounding
for prices, moving averages, contractions, volumes, and pivots.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NumericSpecification:
    field_name: str
    representation: str
    precision_decimals: int
    min_value: Optional[float]
    max_value: Optional[float]
    rounding_mode: str
    description: str


NUMERIC_SPECS: List[NumericSpecification] = [
    NumericSpecification(
        field_name="price",
        representation="IEEE 754 64-bit float / Decimal",
        precision_decimals=4,
        min_value=0.0001,
        max_value=1_000_000.0,
        rounding_mode="ROUND_HALF_UP_4_DECIMALS",
        description="Daily OHLC prices rounded to 4 decimals.",
    ),
    NumericSpecification(
        field_name="volume",
        representation="64-bit integer",
        precision_decimals=0,
        min_value=0.0,
        max_value=100_000_000_000.0,
        rounding_mode="INTEGER_TRUNCATE",
        description="Daily share volume as non-negative integer.",
    ),
    NumericSpecification(
        field_name="percentage_ratio",
        representation="float ratio [0.0, 1.0]",
        precision_decimals=4,
        min_value=0.0,
        max_value=100.0,
        rounding_mode="ROUND_HALF_UP_4_DECIMALS",
        description="Percentage values represented as fractional decimals (e.g., 0.1500 = 15.00%).",
    ),
    NumericSpecification(
        field_name="moving_average",
        representation="float",
        precision_decimals=4,
        min_value=0.0001,
        max_value=1_000_000.0,
        rounding_mode="ROUND_HALF_UP_4_DECIMALS",
        description="Simple Moving Average over N closed daily sessions.",
    ),
    NumericSpecification(
        field_name="contraction_depth",
        representation="float ratio (High - Low) / High",
        precision_decimals=4,
        min_value=0.0,
        max_value=1.0,
        rounding_mode="ROUND_HALF_UP_4_DECIMALS",
        description="Percentage depth of contraction wave.",
    ),
    NumericSpecification(
        field_name="volume_dry_up_ratio",
        representation="float ratio final_wave_avg_vol / sma_50_vol",
        precision_decimals=4,
        min_value=0.0,
        max_value=100.0,
        rounding_mode="ROUND_HALF_UP_4_DECIMALS",
        description="Volume relative to 50-day SMA volume.",
    ),
]


class VCPNumericContract:
    """Deterministic numeric calculation engine for VCP domain contracts."""

    CONTRACT_ID = "ARX_VCP_NUMERIC_CONTRACT"
    VERSION = "1.0.0"

    PRICE_DECIMALS = 4
    PERCENT_DECIMALS = 4

    @staticmethod
    def round_price(val: float) -> float:
        return round(float(val), VCPNumericContract.PRICE_DECIMALS)

    @staticmethod
    def round_ratio(val: float) -> float:
        return round(float(val), VCPNumericContract.PERCENT_DECIMALS)

    @staticmethod
    def compute_sma(values: Sequence[float], period: int) -> Optional[float]:
        """Calculates Simple Moving Average over the trailing `period` closed sessions."""
        if len(values) < period:
            return None
        window = values[-period:]
        avg = sum(window) / float(period)
        return VCPNumericContract.round_price(avg)

    @staticmethod
    def compute_sma_series(values: Sequence[float], period: int) -> List[Optional[float]]:
        """Calculates trailing SMA series matching input length."""
        result: List[Optional[float]] = []
        for i in range(len(values)):
            if i + 1 < period:
                result.append(None)
            else:
                window = values[i + 1 - period : i + 1]
                avg = sum(window) / float(period)
                result.append(VCPNumericContract.round_price(avg))
        return result

    @staticmethod
    def compute_sma_slope(
        sma_series: Sequence[Optional[float]], lookback_bars: int = 22
    ) -> Optional[float]:
        """Calculates percentage change of SMA over trailing lookback_bars."""
        if not sma_series:
            return None
        current = sma_series[-1]
        if current is None:
            return None
        if len(sma_series) <= lookback_bars:
            prior = next((x for x in sma_series if x is not None), None)
        else:
            prior = sma_series[-(lookback_bars + 1)]
            if prior is None:
                prior = next((x for x in sma_series if x is not None), None)

        if prior is None or prior <= 0:
            return 0.0  # Default non-declining baseline if only 1 bar exists
        slope = (current - prior) / prior
        return VCPNumericContract.round_ratio(slope)

    @staticmethod
    def compute_contraction_depth(high: float, low: float) -> float:
        """Calculates contraction depth: (high - low) / high."""
        if high <= 0 or low > high:
            return 0.0
        depth = (high - low) / high
        return VCPNumericContract.round_ratio(depth)

    @staticmethod
    def verify_progressive_tightening(depths: Sequence[float]) -> bool:
        """Verifies strict monotonic contraction depth sequence: Depth_k < Depth_{k-1}."""
        if len(depths) < 2:
            return False
        for i in range(1, len(depths)):
            # Strictly decreasing
            if depths[i] >= depths[i - 1]:
                return False
        return True

    @staticmethod
    def compute_volume_ratio(
        wave_volumes: Sequence[float], baseline_volume_sma: float
    ) -> Optional[float]:
        """Calculates average volume of wave divided by baseline volume SMA."""
        if not wave_volumes or baseline_volume_sma <= 0:
            return None
        avg_vol = sum(wave_volumes) / float(len(wave_volumes))
        ratio = avg_vol / baseline_volume_sma
        return VCPNumericContract.round_ratio(ratio)

    @staticmethod
    def compute_pivot_distance(current_price: float, pivot_price: float) -> float:
        """Calculates distance of current price to pivot: (current - pivot) / pivot."""
        if pivot_price <= 0:
            return 0.0
        dist = (current_price - pivot_price) / pivot_price
        return VCPNumericContract.round_ratio(dist)

    def compute_contract_hash(self) -> str:
        serialized = {
            "contract_id": self.CONTRACT_ID,
            "version": self.VERSION,
            "price_decimals": self.PRICE_DECIMALS,
            "percent_decimals": self.PERCENT_DECIMALS,
            "specs": [asdict(s) for s in sorted(NUMERIC_SPECS, key=lambda x: x.field_name)],
        }
        data_bytes = json.dumps(serialized, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()

    def export_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.CONTRACT_ID,
            "version": self.VERSION,
            "hash": self.compute_contract_hash(),
            "specifications": [asdict(s) for s in NUMERIC_SPECS],
        }
