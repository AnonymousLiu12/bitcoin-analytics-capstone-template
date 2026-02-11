import logging
from pathlib import Path

import pandas as pd

from template.backtest_template import run_full_analysis
from template.prelude_template import load_data

from example_2.model_development_example_2 import (
    compute_window_weights,
    precompute_features,
)

_FEATURES_DF = None


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Adapter for template backtest engine."""
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Starting Bitcoin DCA Strategy Analysis - Example 2 (2-Week 20% Dip)")

    btc_df = load_data()
    _FEATURES_DF = precompute_features(btc_df)

    output_dir = Path(__file__).parent / "output"
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="Example 2 (2-Week 20% Dip)",
    )


if __name__ == "__main__":
    main()
