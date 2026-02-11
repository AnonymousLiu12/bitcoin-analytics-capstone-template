import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Robustly determine the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')
COINMETRICS_PATH = os.path.join(DATA_DIR, 'Coin Metrics', 'coinmetrics_btc.csv')
POLYMARKET_DIR = os.path.join(DATA_DIR, 'Polymarket')

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# --- Data Loading ---
def load_bitcoin_data(filepath):
    print(f"Loading Bitcoin data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        print(f"Error loading Bitcoin data: {e}")
        return None

def load_polymarket_data(datadir):
    print(f"Loading Polymarket data from {datadir}...")
    markets_path = os.path.join(datadir, 'finance_politics_markets.parquet')
    odds_path = os.path.join(datadir, 'finance_politics_odds_history.parquet')
    summary_path = os.path.join(datadir, 'finance_politics_summary.parquet')
    
    data = {}
    try:
        if os.path.exists(markets_path):
            markets_df = pd.read_parquet(markets_path)
            # Convert date columns
            if 'created_at' in markets_df.columns:
                markets_df['created_at'] = pd.to_datetime(markets_df['created_at'])
            if 'end_date' in markets_df.columns:
                markets_df['end_date'] = pd.to_datetime(markets_df['end_date'])
            data['markets'] = markets_df
            print(f"Loaded {len(markets_df)} markets.")
            
        if os.path.exists(odds_path):
            data['odds'] = pd.read_parquet(odds_path)
            print(f"Loaded {len(data['odds'])} odds history records.")
            
        if os.path.exists(summary_path):
            data['summary'] = pd.read_parquet(summary_path)
            print(f"Loaded {len(data['summary'])} summary records.")
            
        return data if data else None
    except Exception as e:
        print(f"Error loading Polymarket data: {e}")
        return None

# --- Analysis & Visualization ---

def plot_btc_price(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['PriceUSD'], label='BTC Price (USD)')
    plt.title('Bitcoin Price History')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'btc_price_history.png'))
    print("Saved btc_price_history.png")
    plt.close()

def plot_polymarket_volume(df):
    plt.figure(figsize=(10, 6))
    # Filter for top 10 categories by volume
    if 'volume' in df.columns and 'category' in df.columns:
        top_cats = df.groupby('category')['volume'].sum().sort_values(ascending=False).head(10)
        sns.barplot(x=top_cats.values, y=top_cats.index)
        plt.title('Top 10 Polymarket Categories by Volume')
        plt.xlabel('Total Volume')
        plt.ylabel('Category')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'polymarket_volume_by_category.png'))
        print("Saved polymarket_volume_by_category.png")
    else:
        print("Columns 'volume' or 'category' not found in Polymarket data.")
    plt.close()

def analyze_btc_metrics(df):
    print("\n--- Bitcoin Data Summary ---")
    print(df[['PriceUSD', 'CapMrktCurUSD', 'HashRate']].describe())
    
    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[['PriceUSD', 'CapMrktCurUSD', 'HashRate', 'TxCnt']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation of Bitcoin Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'btc_correlation_matrix.png'))
    print("Saved btc_correlation_matrix.png")
    plt.close()

def analyze_polymarket_summary(data):
    print("\n--- Polymarket Data Summary ---")
    
    markets_df = data.get('markets')
    if markets_df is not None:
        print(f"Total Markets: {len(markets_df)}")
        if 'active' in markets_df.columns:
            print(f"Active Markets: {markets_df['active'].sum()}")
            print(f"Closed Markets: {len(markets_df) - markets_df['active'].sum()}")
        
        if 'volume' in markets_df.columns:
            print(f"Total Volume: {markets_df['volume'].sum():,.2f}")
            print(f"Average Volume per Market: {markets_df['volume'].mean():,.2f}")

    odds_df = data.get('odds')
    if odds_df is not None:
        print(f"Total Odds History Records: {len(odds_df):,}")
        
    summary_df = data.get('summary')
    if summary_df is not None and 'trade_count' in summary_df.columns:
        print(f"Total Trades: {summary_df['trade_count'].sum():,}")

# --- Main Execution ---
def main():
    btc_df = load_bitcoin_data(COINMETRICS_PATH)
    poly_data = load_polymarket_data(POLYMARKET_DIR)

    if btc_df is not None:
        analyze_btc_metrics(btc_df)
        plot_btc_price(btc_df)
    
    if poly_data is not None:
        analyze_polymarket_summary(poly_data)
        if 'markets' in poly_data:
            plot_polymarket_volume(poly_data['markets'])

    print("\nEDA Layout Complete. Check the 'plots' directory for visualizations.")

if __name__ == "__main__":
    main()
