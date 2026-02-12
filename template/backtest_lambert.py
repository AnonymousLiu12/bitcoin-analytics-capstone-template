import logging
from pathlib import Path

import pandas as pd

try:
    from template.backtest_template import run_full_analysis
    from template.model_development_lambert import (
        compute_window_weights,
        precompute_features,
    )
    from template.prelude_template import load_data
except ImportError:
    from backtest_template import run_full_analysis
    from model_development_lambert import compute_window_weights, precompute_features
    from prelude_template import load_data

# Global variable to store precomputed features
_FEATURES_DF = None


def compute_weights_modal(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper using Lambert strategy compute_window_weights for backtest."""
    global _FEATURES_DF

    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")

    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    current_date = end_date

    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)


def main():
    global _FEATURES_DF

    logging.info("Starting Lambert Strategy Analysis")
    btc_df = load_data()

    logging.info("Precomputing Lambert features...")
    _FEATURES_DF = precompute_features(btc_df)

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "output_lambert"

    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_modal,
        output_dir=output_dir,
        strategy_label="Stablecoins",
    )


if __name__ == "__main__":
    main()

