# Stacking Sats: Improving Bitcoin Accumulation 

Building and improving data-driven Bitcoin accumulation strategies, with a focus on utilizing signal from predicion market data. 

See [stackingsats.org](https://www.stackingsats.org/) for more information.

---

## The Mission: Improving Institutional Bitcoin Accumulation

As Bitcoin matures as an institutional asset, standard Dollar Cost Averaging (DCA) can be suboptimal. Our goal is to design **data-driven, long-only** accumulation strategies that maintain DCA’s systematic discipline while **improving acquisition efficiency** within fixed budgets and time horizons.

### Latest Tournament
Trilemma Foundation hosts tournaments to find the most efficient accumulation models.
* **Current/Recent:** [Stacking Sats Tournament - MSTR 2025](https://github.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025)

---

## Repository Overview

This repository provides a template and framework for:
1.  **Exploratory Data Analysis (EDA)** of Bitcoin price action and on-chain properties.
2.  **Feature Engineering** that integrates prediction market sentiment (Polymarket), macro indicators, and on-chain metrics.
3.  **Strategy Development** for daily purchase schedules (dynamic DCA).
4.  **Backtesting & Evaluation** against uniform DCA benchmarks.

### Repository Structure

```text
.
├── template/                        # CORE FRAMEWORK (Start here)
│   ├── prelude_template.py          # Data loading & Polymarket utilities
│   ├── model_development_template.py # IMPLEMENT YOUR MODEL LOGIC HERE
│   ├── backtest_template.py         # Evaluation engine
│   └── *.md                         # Documentation for model logic & backtesting
├── example_1/                       # REFERENCE IMPLEMENTATION
│   ├── run_backtest.py              # How to run the example
│   └── model_development_example_1.py # Example Polymarket + MVRV integration
├── data/                            # Bitcoin & Polymarket source data
├── output/                          # Results and visualizations
└── tests/                           # Unit tests for core logic
```

---

## Getting Started

### 1. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TrilemmaFoundation/bitcoin-analytics-capstone-template
    cd bitcoin-analytics-capstone-template
    ```

2.  **Setup environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\\Scripts\\activate
    pip install -r requirements.txt
    ```

### 2. Data Acquisition

The `data/` directory contains historical BTC price data and specific Polymarket datasets (Politics, Finance, Crypto).

Data can be [downloaded manually from Google Drive](https://drive.google.com/drive/folders/1gizJ_n-QCnE8qrFM-BU3J_ZpaR3HCjn7?usp=sharing) into the `data/` folder, or you can use the automated script:

```bash
python data/download_data.py
```

**Included Data:**
* **CoinMetrics BTC Data**: Daily OHLCV and network metrics.
* **Polymarket Data**: High-fidelity parquet files containing trades, odds history, and market metadata.

---

## Building Your Model

The framework uses a **Template Baseline** located in `template/`. This model uses a simple 200-day Moving Average filter: buy more when price is below the 200-day MA, buy less when above.

### The Challenge: Prediction Market Integration

The primary objective is to evolve this baseline into a market-aware strategy by leveraging **Polymarket data**.

**Potential Signal Leads:**
* **Election Probabilities**: Does political volatility lead BTC price discovery?
* **Economic Indicators**: Do prediction markets for Fed rate cuts lead BTC price movements?
* **Retail Sentiment**: Using specific "Polymarket Crypto" markets as indicators for retail exuberance.

### Running Backtests

**Baseline Model:**
```bash
python -m template.backtest_template
```

**Reference Implementation (Example 1):**
```bash
python -m example_1.run_backtest
```

---

## Evaluation Metrics

Strategies are evaluated on three primary pillars (automated via the backtest engine):

1.  **Win Rate**: The percentage of 1-year windows where the strategy outperforms uniform DCA.
2.  **SPD (Sats Per Dollar)**: The overall efficiency of satoshi accumulation compared to the baseline.
3.  **Model Score**: A composite metric weighing win rate against reward-to-risk percentile.

## Licensing

*   **Code:** This repository, including its analysis and documentation, is open-sourced under the **MIT License**.
*   **Data:** The data provided (e.g., CoinMetrics, Polymarket) is not covered by the MIT license and retains its original licensing terms. Please refer to the respective data providers for their terms of use.

---

## Contacts & Community

* **App:** [stackingsats.org](https://www.stackingsats.org/)
* **Website:** [trilemma.foundation](https://www.trilemma.foundation/)
* **Foundation:** [Trilemma Foundation](https://github.com/TrilemmaFoundation)
