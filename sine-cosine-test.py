import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Import project modules
from src.model import TransformerModel, BaselineModel
from src.process_data import create_sequences
from src.backtest import Backtester

def generate_synthetic_data(wave_type='sine', n_periods=20, points_per_period=100, noise_level=0.1, start_date='2023-01-01'):
    """
    Generate synthetic price data based on sine or cosine waves
    
    Args:
        wave_type: 'sine', 'cosine', or 'both' (combines both waves)
        n_periods: Number of complete wave cycles
        points_per_period: Number of data points per cycle
        noise_level: Standard deviation of Gaussian noise to add
        start_date: Starting date for the time series
        
    Returns:
        DataFrame with synthetic price data
    """
    n_points = n_periods * points_per_period
    x = np.linspace(0, n_periods * 2 * np.pi, n_points)
    
    # Generate base wave
    if wave_type == 'sine':
        y = np.sin(x)
    elif wave_type == 'cosine':
        y = np.cos(x)
    elif wave_type == 'both':
        y = np.sin(x) + 0.5 * np.cos(2 * x)  # Combine sine with a faster cosine
    else:
        raise ValueError(f"Unsupported wave_type: {wave_type}")
    
    # Add some noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, n_points)
        y += noise
    
    # Scale to reasonable price range (e.g., Bitcoin-like prices)
    y = (y + 2) * 10000  # Scale to range approximately 10000-30000
    
    # Create date range
    start_date = pd.Timestamp(start_date)
    date_range = [start_date + timedelta(hours=i) for i in range(n_points)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': y,
        'high': y * 1.01,  # Simulate slightly higher highs
        'low': y * 0.99,   # Simulate slightly lower lows
        'close': y,
        'volume': np.random.randn(n_points) * 100 + 1000  # Random volume
    }, index=date_range)
    
    # Add returns column
    df['returns'] = df['close'].pct_change()
    
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame
    Simplified version of the function in process_data.py
    """
    df_features = df.copy()
    
    # Price features
    df_features['ma7'] = df['close'].rolling(window=7).mean()
    df_features['ma14'] = df['close'].rolling(window=14).mean()
    df_features['ma30'] = df['close'].rolling(window=30).mean()
    
    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df_features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df_features['macd'] = ema12 - ema26
    df_features['macd_signal'] = df_features['macd'].ewm(span=9).mean()
    
    # Volatility
    df_features['volatility'] = df['returns'].rolling(window=14).std()
    
    # Momentum
    df_features['momentum'] = df['close'] / df['close'].shift(4) - 1
    
    # Clean up NaN values
    df_features = df_features.dropna()
    
    return df_features

def generate_signals_from_model(model, features, device='cpu', threshold=0.0001):
    """
    Same as in backtest_testset_only.py
    """
    model = model.to(device)
    model.eval()
    signals = []
    
    with torch.no_grad():
        for i in range(len(features)):
            try:
                x = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
                output = model(x)
                signal = output.cpu().numpy()[0][0]
                
                if abs(signal) < threshold:
                    signal_value = 0
                else:
                    signal_value = np.sign(signal)
                
                signals.append(signal_value)
            except Exception as e:
                print(f"Error at step {i}: {e}")
                signals.append(0.0)
    
    signal_array = np.array(signals)
    buy_count = np.sum(signal_array == 1)
    sell_count = np.sum(signal_array == -1)
    hold_count = np.sum(signal_array == 0)
    total = len(signal_array)
    
    print("\nSignal Distribution:")
    print(f"  Buy signals: {buy_count} ({buy_count/total*100:.2f}%)")
    print(f"  Sell signals: {sell_count} ({sell_count/total*100:.2f}%)")
    print(f"  Hold signals: {hold_count} ({hold_count/total*100:.2f}%)")
    
    return signal_array

def load_experiment(experiment_dir):
    """Load model and parameters from experiment directory"""
    # Load parameters
    with open(os.path.join(experiment_dir, 'params.txt'), 'r') as f:
        params = json.load(f)
    
    # Load scaler if available
    scaler_path = os.path.join(experiment_dir, 'scaler.pt')
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = torch.load(scaler_path, weights_only=False)
        except Exception as e:
            print(f"Warning: Could not load scaler: {e}")
            print("Proceeding without feature scaling")
    
    return params, scaler

def run_test_on_sine_wave(experiment_dir, wave_type='sine', n_periods=20, 
                          points_per_period=100, noise_level=0.1, 
                          commission=0.001, initial_capital=10000, 
                          compare_baseline=True, threshold=0.0001, force_cpu=False):
    """
    Test a trained model on synthetic sine/cosine wave data
    """
    # Load experiment data
    params, scaler = load_experiment(experiment_dir)
    
    # Generate synthetic data
    print(f"Generating synthetic {wave_type} wave data with {n_periods} periods")
    print(f"Noise level: {noise_level}")
    df = generate_synthetic_data(
        wave_type=wave_type, 
        n_periods=n_periods, 
        points_per_period=points_per_period,
        noise_level=noise_level
    )
    
    # Add technical indicators
    print("Adding technical indicators")
    df_features = add_technical_indicators(df)
    
    # Create sequences
    seq_length = params.get('seq_length', 48)
    feature_cols = [col for col in df_features.columns if col != 'returns']
    print(f"Creating sequences with length {seq_length}")
    X, y = create_sequences(df_features, seq_length, 'returns', feature_cols)
    
    print(f"Total data shape: X={X.shape}")
    
    # Split the data - using 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"Test data shape: X_test={X_test.shape}")
    
    # Calculate the starting index for the test set in the original data
    test_start_idx = len(X) - len(X_test)
    test_end_idx = len(X)
    
    # Add sequence length to get the actual index in the processed DataFrame
    df_test_start_idx = test_start_idx + seq_length
    
    # Display date range of test set
    df_index_array = df_features.index.to_numpy()
    test_start_date = pd.Timestamp(df_index_array[df_test_start_idx])
    test_end_date = pd.Timestamp(df_index_array[df_test_start_idx + len(X_test) - 1])
    
    print(f"Test set date range: {test_start_date.strftime('%Y-%m-%d %H:%M:%S')} to {test_end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate test period duration in days
    test_duration_days = (test_end_date - test_start_date).days + (test_end_date - test_start_date).seconds / (60 * 60 * 24)
    print(f"Test period duration: {test_duration_days:.1f} days")
    
    # Apply scaling to test data if needed
    if scaler is not None:
        print("Applying feature scaling")
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_reshaped = scaler.transform(X_test_reshaped)
        X_test = X_test_reshaped.reshape(X_test.shape)
    
    # Set device for model
    if force_cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print(f"Using device: {device}")
    
    # Load model
    input_dim = X_test.shape[-1]
    hidden_dim = params.get('hidden_dim', 64)
    num_heads = params.get('num_heads', 4)
    num_layers = params.get('num_layers', 2)
    dropout = params.get('dropout', 0.1)
    
    print(f"Loading model from {experiment_dir}")
    model = TransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Load state dict with appropriate device mapping
    model_path = os.path.join(experiment_dir, 'model.pt')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Also load baseline model for comparison if requested
    if compare_baseline:
        print("Creating baseline model for comparison")
        baseline_model = BaselineModel(window_size=20)
        baseline_model = baseline_model.to(device)
    
    # Generate signals
    print(f"Generating trading signals with threshold {threshold}")
    signals = generate_signals_from_model(model, X_test, device=device, threshold=threshold)
    
    # Also generate baseline signals if requested
    if compare_baseline:
        print("\nGenerating baseline model signals:")
        baseline_signals = generate_signals_from_model(baseline_model, X_test, device=device)
    
    # Create a subset of the price data that matches the test set
    test_df = df_features.iloc[df_test_start_idx:df_test_start_idx + len(X_test)].copy()
    
    # Run backtest
    print(f"Running backtest with {commission:.2%} commission on test data")
    
    # Create backtester
    backtester = Backtester(
        price_data=test_df,
        commission=commission,
        initial_capital=initial_capital
    )
    
    # Run backtest on transformer model
    transformer_results = backtester.backtest(signals, start_idx=0)
    
    # Run backtest on baseline model if requested
    if compare_baseline:
        baseline_results = backtester.backtest(baseline_signals, start_idx=0)
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # Plot the original wave first
    ax0 = plt.subplot(4, 1, 1)
    plt.plot(test_df.index, test_df['close'], label=f'{wave_type.capitalize()} Wave')
    plt.title(f'Synthetic {wave_type.capitalize()} Wave Data')
    plt.grid(True)
    plt.legend()
    
    # Plot portfolio value
    ax1 = plt.subplot(4, 1, 2)
    
    # Fix: Ensure transformer_values and test_df indices have the same length
    transformer_values = np.array(transformer_results['portfolio_values'])
    if len(transformer_values) != len(test_df):
        transformer_values = transformer_values[1:]  # Skip the initial value
    
    # Use percentile for better scaling
    y_max = np.percentile(transformer_values[np.isfinite(transformer_values)], 99)
    
    plt.plot(test_df.index, transformer_values, label='Transformer')
    
    if compare_baseline:
        baseline_values = np.array(baseline_results['portfolio_values'])
        if len(baseline_values) != len(test_df):
            baseline_values = baseline_values[1:]  # Skip the initial value
        plt.plot(test_df.index, baseline_values, label='Baseline SMA')
        y_max = max(y_max, np.percentile(baseline_values[np.isfinite(baseline_values)], 99))
    
    plt.plot(test_df.index, [initial_capital] * len(test_df), 'k--', label='Initial Capital')
    plt.ylim(0, y_max * 1.1)  # Set y-axis limit for better visualization
    plt.title(f'Portfolio Value Over Time (Test Set)')
    plt.grid(True)
    plt.legend()
    
    # Plot positions
    ax2 = plt.subplot(4, 1, 3)
    positions = np.array(transformer_results['positions'])
    if len(positions) != len(test_df):
        positions = positions[1:]  # Skip the initial position
    plt.plot(test_df.index, positions, label='Transformer')
    
    if compare_baseline:
        baseline_positions = np.array(baseline_results['positions'])
        if len(baseline_positions) != len(test_df):
            baseline_positions = baseline_positions[1:]  # Skip the initial position
        plt.plot(test_df.index, baseline_positions, label='Baseline SMA')
    
    plt.title('Position Over Time (-1: Short, 0: None, 1: Long)')
    plt.grid(True)
    plt.legend()
    
    # Plot equity curve vs buy and hold
    ax3 = plt.subplot(4, 1, 4)
    
    # Calculate buy and hold returns (using only test data)
    prices = test_df['close'].values
    initial_price = prices[0]
    buy_hold_values = [initial_capital * (price / initial_price) for price in prices]
    
    # Calculate normalized returns with sanity checking
    def safe_normalize(values, initial):
        norm_vals = []
        for val in values:
            if np.isfinite(val):
                pct = (val / initial * 100) - 100
                pct = min(pct, 300)  # Cap for visualization
                norm_vals.append(pct)
            else:
                norm_vals.append(norm_vals[-1] if norm_vals else 0)
        return norm_vals
    
    transformer_returns = safe_normalize(transformer_values, initial_capital)
    buy_hold_returns = safe_normalize(buy_hold_values, initial_capital)
    
    if compare_baseline:
        baseline_returns = safe_normalize(baseline_values, initial_capital)
    
    plt.plot(test_df.index, transformer_returns, label='Transformer')
    if compare_baseline:
        plt.plot(test_df.index, baseline_returns, label='Baseline SMA')
    plt.plot(test_df.index, buy_hold_returns, label='Buy & Hold')
    plt.title('Percentage Returns vs Buy & Hold (Test Set Only)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save results
    results_dir = f'sine_wave_test_{wave_type}'
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, f'{wave_type}_wave_test_results.png'))
    print(f"\nResults saved to {results_dir}")
    
    # Print summary metrics
    print("\n============= SINE WAVE TEST RESULTS =============")
    print(f"Wave type: {wave_type}")
    print(f"Noise level: {noise_level}")
    print(f"Number of periods: {n_periods}")
    
    print("\n============= TRANSFORMER MODEL =============")
    print(f"Final Value: ${transformer_results['final_value']:.2f}")
    print(f"Total Return: {transformer_results['total_return']:.2f}%")
    print(f"Sharpe Ratio: {transformer_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {transformer_results['max_drawdown']:.2f}%")
    print(f"Number of Trades: {len(transformer_results['trades'])}")
    
    if transformer_results['trades']:
        winning_trades = sum(1 for trade in transformer_results['trades'] if trade['profit_pct'] > 0)
        win_rate = winning_trades / len(transformer_results['trades']) * 100
        print(f"Win Rate: {win_rate:.2f}%")
    
    if compare_baseline:
        print("\n============= BASELINE MODEL =============")
        print(f"Final Value: ${baseline_results['final_value']:.2f}")
        print(f"Total Return: {baseline_results['total_return']:.2f}%")
        print(f"Sharpe Ratio: {baseline_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {baseline_results['max_drawdown']:.2f}%")
        print(f"Number of Trades: {len(baseline_results['trades'])}")
        
        if baseline_results['trades']:
            winning_trades = sum(1 for trade in baseline_results['trades'] if trade['profit_pct'] > 0)
            win_rate = winning_trades / len(baseline_results['trades']) * 100
            print(f"Win Rate: {win_rate:.2f}%")
    
    # Calculate buy and hold metrics
    buy_hold_return = (buy_hold_values[-1] / buy_hold_values[0] - 1) * 100
    
    print("\n============= BUY & HOLD =============")
    print(f"Final Value: ${buy_hold_values[-1]:.2f}")
    print(f"Total Return: {buy_hold_return:.2f}%")
    
    # Save results to file
    results_path = os.path.join(results_dir, f'{wave_type}_wave_test_results.txt')
    with open(results_path, 'w') as f:
        f.write("============= SINE WAVE TEST RESULTS =============\n")
        f.write(f"Wave type: {wave_type}\n")
        f.write(f"Noise level: {noise_level}\n")
        f.write(f"Number of periods: {n_periods}\n\n")
        
        f.write("============= TRANSFORMER MODEL =============\n")
        f.write(f"Final Value: ${transformer_results['final_value']:.2f}\n")
        f.write(f"Total Return: {transformer_results['total_return']:.2f}%\n")
        f.write(f"Sharpe Ratio: {transformer_results['sharpe_ratio']:.2f}\n")
        f.write(f"Max Drawdown: {transformer_results['max_drawdown']:.2f}%\n")
        f.write(f"Number of Trades: {len(transformer_results['trades'])}\n")
        
        if transformer_results['trades']:
            f.write(f"Win Rate: {win_rate:.2f}%\n\n")
        
        if compare_baseline:
            f.write("============= BASELINE MODEL =============\n")
            f.write(f"Final Value: ${baseline_results['final_value']:.2f}\n")
            f.write(f"Total Return: {baseline_results['total_return']:.2f}%\n")
            f.write(f"Sharpe Ratio: {baseline_results['sharpe_ratio']:.2f}\n")
            f.write(f"Max Drawdown: {baseline_results['max_drawdown']:.2f}%\n")
            f.write(f"Number of Trades: {len(baseline_results['trades'])}\n")
            
            if baseline_results['trades']:
                winning_trades = sum(1 for trade in baseline_results['trades'] if trade['profit_pct'] > 0)
                win_rate = winning_trades / len(baseline_results['trades']) * 100
                f.write(f"Win Rate: {win_rate:.2f}%\n\n")
        
        f.write("============= BUY & HOLD =============\n")
        f.write(f"Final Value: ${buy_hold_values[-1]:.2f}\n")
        f.write(f"Total Return: {buy_hold_return:.2f}%\n")
    
    print(f"\nResults saved to {results_path}")
    
    return transformer_results, baseline_results if compare_baseline else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a trained transformer model on synthetic sine/cosine wave data')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Path to the experiment directory')
    parser.add_argument('--wave-type', type=str, default='sine', choices=['sine', 'cosine', 'both'],
                        help='Type of wave to generate (sine, cosine, or both)')
    parser.add_argument('--periods', type=int, default=20,
                        help='Number of complete wave cycles')
    parser.add_argument('--points-per-period', type=int, default=100,
                        help='Number of data points per wave cycle')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Amount of noise to add (standard deviation)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Trading commission as a decimal (default: 0.001 = 0.1%)')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital for the backtest (default: 10000)')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Disable baseline model comparison')
    parser.add_argument('--threshold', type=float, default=0.0001,
                        help='Threshold for signal generation (default: 0.0001)')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    
    args = parser.parse_args()
    
    run_test_on_sine_wave(
        experiment_dir=args.experiment,
        wave_type=args.wave_type,
        n_periods=args.periods,
        points_per_period=args.points_per_period,
        noise_level=args.noise,
        commission=args.commission,
        initial_capital=args.capital,
        compare_baseline=not args.no_baseline,
        threshold=args.threshold,
        force_cpu=args.force_cpu
    )