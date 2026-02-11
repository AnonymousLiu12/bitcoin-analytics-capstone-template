"""Dynamic DCA weight computation using a 2-week dip trigger.

Single rule:
- If BTC falls by 20% or more over the last 14 days, increase buy weight.
"""

import numpy as np
import pandas as pd

from template.model_development_template import (
    _clean_array,
    allocate_sequential_stable,
)

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"

# Strategy parameters
DIP_LOOKBACK_DAYS = 30
DIP_THRESHOLD = -0.20
DIP_BOOST_MULTIPLIER = 8.0

# Feature column names (for compatibility)
FEATS = ["ret_14d"]


# =============================================================================
# Feature Engineering
# =============================================================================


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 14-day return feature used by the dip trigger strategy."""
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    price = df[PRICE_COL].loc["2010-07-18":].copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        ret_14d = price.pct_change(DIP_LOOKBACK_DAYS).clip(-1.0, 1.0)

    features = pd.DataFrame({PRICE_COL: price, "ret_14d": ret_14d}, index=price.index)
    # Lag by one day to avoid look-ahead bias.
    features["ret_14d"] = features["ret_14d"].shift(1).fillna(0.0)
    return features.fillna(0.0)


# =============================================================================
# Dynamic Multiplier
# =============================================================================


def compute_dynamic_multiplier(ret_14d: np.ndarray) -> np.ndarray:
    """Boost allocation only when 14-day return is <= -20%."""
    dip_trigger = ret_14d <= DIP_THRESHOLD
    multiplier = np.where(dip_trigger, DIP_BOOST_MULTIPLIER, 1.0)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


# =============================================================================
# Weight Computation API
# =============================================================================


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute weights for one window using precomputed features."""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    ret_14d = _clean_array(df["ret_14d"].values)
    dyn = compute_dynamic_multiplier(ret_14d=ret_14d)
    raw = base * dyn

    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    """Compute lock-on-compute weights for a date window."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Extend feature frame for future dates in production mode.
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
        n_past=n_past,
        locked_weights=locked_weights,
    )
    return weights.reindex(full_range, fill_value=0.0)
