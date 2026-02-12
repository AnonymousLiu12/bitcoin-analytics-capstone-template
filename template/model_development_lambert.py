"""Lambert strategy skeleton for Dynamic DCA.

Copy this file and only change the TODO sections to implement your strategy.
Keep function signatures unchanged so backtest_template can call it directly.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MIN_W = 1e-6
STABLE_START_DATE = "2020-01-01"

FEATS = [
    "stable_roc_30",
    "price_z_90",
]


def _load_stablecoins_series(target_index: pd.Index) -> pd.Series:
    """Load stablecoins market cap and align to BTC date index."""
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "stablecoins.csv"
    if not csv_path.exists():
        csv_path = Path("stablecoins.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"stablecoins.csv not found at {csv_path}. "
            "Place stablecoins.csv at project root."
        )

    stable_df = pd.read_csv(csv_path)
    required_cols = {"date", "stable_mcap"}
    if not required_cols.issubset(stable_df.columns):
        raise ValueError(
            f"stablecoins.csv must contain columns {required_cols}, got {set(stable_df.columns)}"
        )

    stable_df["date"] = pd.to_datetime(stable_df["date"])
    stable_df = stable_df.set_index("date").sort_index()
    stable_df.index = stable_df.index.normalize().tz_localize(None)
    stable_df = stable_df.loc[stable_df.index >= pd.to_datetime(STABLE_START_DATE)]

    stable = pd.to_numeric(stable_df["stable_mcap"], errors="coerce")
    stable = stable.reindex(target_index).ffill().bfill()
    return stable


# =============================================================================
# Feature Engineering
# =============================================================================


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute strategy features once for fast backtesting.

    Rules:
    - Do not use future data.
    - Lag predictive features by 1 day (shift(1)).
    - Keep PRICE_COL in the returned DataFrame.
    """
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    price = df[PRICE_COL].copy()

    stable_mcap = _load_stablecoins_series(price.index)

    # 1) Stablecoin fuel gauge: 30-day Rate of Change.
    with np.errstate(divide="ignore", invalid="ignore"):
        stable_roc_30 = (stable_mcap / stable_mcap.shift(30)) - 1.0
    stable_roc_30 = stable_roc_30.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 2) Price level gauge: 90-day Z-score.
    price_ma_90 = price.rolling(window=90, min_periods=30).mean()
    price_std_90 = price.rolling(window=90, min_periods=30).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_z_90 = (price - price_ma_90) / price_std_90
    price_z_90 = price_z_90.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "stable_mcap": stable_mcap,
            # Lag both predictive features by one day to prevent look-ahead bias.
            "stable_roc_30": stable_roc_30.shift(1).fillna(0.0),
            "price_z_90": price_z_90.shift(1).fillna(0.0),
        },
        index=price.index,
    )
    return features


# =============================================================================
# Strategy Mapping: feature -> multiplier
# =============================================================================


def compute_dynamic_multiplier(
    stable_roc_30: np.ndarray, price_z_90: np.ndarray
) -> np.ndarray:
    """Rule-based multiplier using stablecoin ROC and price Z-score.

    Rules (tuned):
    - A+ extreme dip buy: stable > -2% and z < -1.6 -> 3.0x
    - A normal accumulation: stable > 0% and z < -1.1 -> 1.7x
    - B healthy uptrend: stable > 0% and z >= -1.1 -> 1.0x
    - C weak rally: stable < 0% and z > 1.0 -> 0.8x
    - D death spiral defense: stable < -3% and z < 0 -> 0.4x
    - Else fallback -> 1.0x
    """
    n = min(len(stable_roc_30), len(price_z_90))
    if n == 0:
        return np.array([])

    roc = stable_roc_30[:n]
    z = price_z_90[:n]
    mult = np.ones(n, dtype=float)

    mask_a_plus = (roc > -0.02) & (z < -1.6)
    mask_a = (roc > 0.0) & (z < -1.1)
    mask_b = (roc > 0.0) & (z >= -1.1)
    mask_c = (roc < 0.0) & (z > 1.0)
    mask_d = (roc < -0.03) & (z < 0.0)

    # Apply in priority order.
    mult[mask_b] = 1.0
    mult[mask_c] = 0.8
    mult[mask_d] = 0.4
    mult[mask_a] = 1.7
    mult[mask_a_plus] = 3.0

    return np.where(np.isfinite(mult), mult, 1.0)


def _clean_array(arr: np.ndarray) -> np.ndarray:
    """Replace non-finite values with zeros."""
    return np.where(np.isfinite(arr), arr, 0.0)


# =============================================================================
# Weight API used by backtest_template
# =============================================================================


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute normalized per-day weights for one window.

    Note:
    - n_past and locked_weights are kept for interface compatibility.
    """
    _ = n_past
    _ = locked_weights

    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n
    stable_roc_30 = _clean_array(df["stable_roc_30"].values)
    price_z_90 = _clean_array(df["price_z_90"].values)
    multiplier = compute_dynamic_multiplier(stable_roc_30, price_z_90)

    raw = base * multiplier
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw = np.clip(raw, MIN_W, None)

    total = raw.sum()
    if total <= 0 or not np.isfinite(total):
        weights = np.ones(n) / n
    else:
        weights = raw / total

    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Public API expected by backtest_template.py."""
    _ = current_date  # Kept for interface compatibility

    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Fill missing dates in feature frame, so returned weights always cover full_range.
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        features_df = pd.concat([features_df, placeholder]).sort_index()

    weights = compute_weights_fast(
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
        n_past=None,
        locked_weights=locked_weights,
    )
    return weights.reindex(full_range, fill_value=0.0)

