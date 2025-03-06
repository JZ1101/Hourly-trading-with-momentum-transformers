import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_data_quality(file_path):
    """Analyze data quality and find potential issues"""
    print(f"Analyzing data file: {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # Basic info
    print("\nBasic Information:")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found.")
    
    # Check for extreme values
    print("\nPrice Range (checking for outliers):")
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            median = df[col].median()
            mean = df[col].mean()
            std = df[col].std()
            print(f"{col}: min={min_val:.2f}, max={max_val:.2f}, median={median:.2f}, mean={mean:.2f}, std={std:.2f}")
            
            # Check for extreme outliers (more than 10 std from mean)
            outliers = df[abs(df[col] - mean) > 10 * std]
            if len(outliers) > 0:
                print(f"  Found {len(outliers)} extreme outliers in {col}")
                print(f"  Example outliers: {outliers[col].head().tolist()}")
    
    # Check for jumps in price
    print("\nChecking for sudden price jumps:")
    df['price_change_pct'] = df['close'].pct_change() * 100
    large_changes = df[abs(df['price_change_pct']) > 10]  # More than 10% change
    if len(large_changes) > 0:
        print(f"Found {len(large_changes)} large price changes (>10%)")
        print("Top 5 largest changes:")
        print(large_changes.sort_values('price_change_pct', ascending=False)[['timestamp', 'close', 'price_change_pct']].head())
    else:
        print("No large price jumps found.")
    
    # Volume analysis
    if 'volume' in df.columns:
        print("\nVolume Analysis:")
        print(f"Min volume: {df['volume'].min()}")
        print(f"Max volume: {df['volume'].max()}")
        print(f"Mean volume: {df['volume'].mean():.2f}")
        
        # Check for zero volume
        zero_volume = df[df['volume'] == 0]
        if len(zero_volume) > 0:
            print(f"Found {len(zero_volume)} periods with zero volume")
    
    # Check for gaps in time series
    print("\nChecking for time gaps:")
    df = df.sort_values('timestamp')
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600  # in hours
    gaps = df[df['time_diff'] > 1.1]  # More than 1.1 hours (allowing for small deviations)
    if len(gaps) > 0:
        print(f"Found {len(gaps)} gaps in time series")
        print("Example gaps:")
        print(gaps[['timestamp', 'time_diff']].head())
    else:
        print("No significant time gaps found.")
    
    # Plot the price data
    plt.figure(figsize=(15, 7))
    plt.plot(df['timestamp'], df['close'])
    plt.title('BTC/USDT Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.grid(True)
    plt.savefig('price_chart.png')
    
    # Plot histogram of returns
    plt.figure(figsize=(12, 6))
    plt.hist(df['price_change_pct'].dropna(), bins=100, alpha=0.75)
    plt.title('Histogram of Hourly Returns (%)')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('returns_histogram.png')
    
    print("\nAnalysis complete. Plots saved as 'price_chart.png' and 'returns_histogram.png'")

if __name__ == "__main__":
    # Change this to your data file path
    file_path = "data/raw/btc_usdt_1h_2020Jan1_2025Mar6.csv"
    check_data_quality(file_path)