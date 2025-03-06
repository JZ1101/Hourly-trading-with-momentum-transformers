import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import project modules
from src.model import TransformerModel, BaselineModel
from src.process_data import load_data, add_technical_indicators, create_sequences
from src.backtest import Backtester, generate_signals_from_model

def load_experiment(experiment_dir):
    """Load model and parameters from experiment directory"""
    # Load parameters
    with open(os.path.join(experiment_dir, 'params.txt'), 'r') as f:
        params = json.load(f)
    
    # Load scaler if available
    scaler_path = os.path.join(experiment_dir, 'scaler.pt')
    scaler = None
    if os.path.exists(scaler_path):
        scaler = torch.load(scaler_path)
    
    return params, scaler

def run_backtest(data_path, experiment_dir, commission=0.001, initial_capital=10000, compare_baseline=True):
    """Run backtest on trained model"""
    # Load experiment data
    params, scaler = load_experiment(experiment_dir)
    
    # Load and process data
    print(f"Loading data from {data_path}")
    df = load_data(data_path)
    
    # Add technical indicators
    print("Adding technical indicators")
    df_features = add_technical_indicators(df)
    
    # Create sequences
    seq_length = params.get('seq_length', 48)
    feature_cols = [col for col in df_features.columns if col != 'returns']
    print(f"Creating sequences with length {seq_length}")
    X, _ = create_sequences(df_features, seq_length, feature_cols=feature_cols)
    
    # Apply scaling if scaler is available
    if scaler is not None:
        print("Applying feature scaling")
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_reshaped = scaler.transform(X_reshaped)
        X = X_reshaped.reshape(X.shape)
    
    # Load model
    input_dim = X.shape[-1]
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
    
    model_path = os.path.join(experiment_dir, 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Also load baseline model for comparison if requested
    if compare_baseline:
        print("Creating baseline model for comparison")
        baseline_model = BaselineModel(window_size=20)
    
    # Generate signals
    print("Generating trading signals")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    signals = generate_signals_from_model(model, X, device=device)
    
    # Also generate baseline signals if requested
    if compare_baseline:
        baseline_signals = generate_signals_from_model(baseline_model, X, device=device)
    
    # Run backtest
    print(f"Running backtest with {commission:.2%} commission")
    
    # Start index (account for sequence length)
    start_idx = seq_length
    
    # Create backtester
    backtester = Backtester(
        price_data=df_features,
        commission=commission,
        initial_capital=initial_capital
    )
    
    # Run backtest on transformer model
    transformer_results = backtester.backtest(signals, start_idx)
    
    # Run backtest on baseline model if requested
    if compare_baseline:
        baseline_results = backtester.backtest(baseline_signals, start_idx)
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # Plot portfolio value
    plt.subplot(3, 1, 1)
    
    # Get valid portfolio values for better visualization
    transformer_values = np.array(transformer_results['portfolio_values'])
    # Use percentile for better scaling
    y_max = np.percentile(transformer_values[np.isfinite(transformer_values)], 99)
    
    plt.plot(transformer_values, label='Transformer')
    if compare_baseline:
        baseline_values = np.array(baseline_results['portfolio_values'])
        plt.plot(baseline_values, label='Baseline SMA')
        y_max = max(y_max, np.percentile(baseline_values[np.isfinite(baseline_values)], 99))
    
    plt.plot([initial_capital] * len(transformer_values), 'k--', label='Initial Capital')
    plt.ylim(0, y_max * 1.1)  # Set y-axis limit for better visualization
    plt.title('Portfolio Value Over Time')
    plt.grid(True)
    plt.legend()
    
    # Plot positions
    plt.subplot(3, 1, 2)
    plt.plot(transformer_results['positions'], label='Transformer')
    if compare_baseline:
        plt.plot(baseline_results['positions'], label='Baseline SMA')
    plt.title('Position Over Time (-1: Short, 0: None, 1: Long)')
    plt.grid(True)
    plt.legend()
    
    # Plot equity curve vs buy and hold
    plt.subplot(3, 1, 3)
    
    # Calculate buy and hold returns
    prices = df_features['close'].iloc[start_idx:start_idx+len(signals)].values
    initial_price = prices[0]
    buy_hold_values = [initial_capital * (price / initial_price) for price in prices]
    
    # Normalize to percentage returns
    max_return_pct = 300  # Cap at 300% for visualization
    
    # Calculate normalized returns with sanity checking
    def safe_normalize(values, initial):
        norm_vals = []
        for val in values:
            if np.isfinite(val):
                pct = (val / initial * 100) - 100
                pct = min(pct, max_return_pct)  # Cap for visualization
                norm_vals.append(pct)
            else:
                norm_vals.append(norm_vals[-1] if norm_vals else 0)
        return norm_vals
    
    transformer_returns = safe_normalize(transformer_results['portfolio_values'], initial_capital)
    buy_hold_returns = safe_normalize(buy_hold_values, initial_capital)
    
    if compare_baseline:
        baseline_returns = safe_normalize(baseline_results['portfolio_values'], initial_capital)
    
    plt.plot(transformer_returns, label='Transformer')
    if compare_baseline:
        plt.plot(baseline_returns, label='Baseline SMA')
    plt.plot(buy_hold_returns, label='Buy & Hold')
    plt.title('Percentage Returns vs Buy & Hold')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'backtest_results.png'))
    
    # Save plot to current directory too for easy access
    plt.savefig('latest_backtest_results.png')
    
    plt.show()
    
    # Print summary metrics
    print("\n============= TRANSFORMER MODEL =============")
    print(f"Final Value: ${transformer_results['final_value']:.2f}")
    print(f"Total Return: {transformer_results['total_return']:.2f}%")
    print(f"Annual Return: {transformer_results['annual_return']:.2f}%")
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
        print(f"Annual Return: {baseline_results['annual_return']:.2f}%")
        print(f"Sharpe Ratio: {baseline_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {baseline_results['max_drawdown']:.2f}%")
        print(f"Number of Trades: {len(baseline_results['trades'])}")
        
        if baseline_results['trades']:
            winning_trades = sum(1 for trade in baseline_results['trades'] if trade['profit_pct'] > 0)
            win_rate = winning_trades / len(baseline_results['trades']) * 100
            print(f"Win Rate: {win_rate:.2f}%")
    
    # Calculate buy and hold metrics
    buy_hold_return = (buy_hold_values[-1] / buy_hold_values[0] - 1) * 100
    buy_hold_annual_return = ((buy_hold_values[-1] / buy_hold_values[0]) ** (1 / (len(buy_hold_values) / (365 * 24))) - 1) * 100
    
    print("\n============= BUY & HOLD =============")
    print(f"Final Value: ${buy_hold_values[-1]:.2f}")
    print(f"Total Return: {buy_hold_return:.2f}%")
    print(f"Annual Return: {buy_hold_annual_return:.2f}%")
    
    # Save results to file
    results_path = os.path.join(experiment_dir, 'backtest_results.txt')
    with open(results_path, 'w') as f:
        f.write("============= TRANSFORMER MODEL =============\n")
        f.write(f"Final Value: ${transformer_results['final_value']:.2f}\n")
        f.write(f"Total Return: {transformer_results['total_return']:.2f}%\n")
        f.write(f"Annual Return: {transformer_results['annual_return']:.2f}%\n")
        f.write(f"Sharpe Ratio: {transformer_results['sharpe_ratio']:.2f}\n")
        f.write(f"Max Drawdown: {transformer_results['max_drawdown']:.2f}%\n")
        f.write(f"Number of Trades: {len(transformer_results['trades'])}\n")
        
        if transformer_results['trades']:
            f.write(f"Win Rate: {win_rate:.2f}%\n\n")
        
        if compare_baseline:
            f.write("============= BASELINE MODEL =============\n")
            f.write(f"Final Value: ${baseline_results['final_value']:.2f}\n")
            f.write(f"Total Return: {baseline_results['total_return']:.2f}%\n")
            f.write(f"Annual Return: {baseline_results['annual_return']:.2f}%\n")
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
        f.write(f"Annual Return: {buy_hold_annual_return:.2f}%\n")
    
    print(f"\nResults saved to {results_path}")
    
    return transformer_results, baseline_results if compare_baseline else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest a trained transformer model')
    parser.add_argument('--data', type=str, default='data/raw/btc_usdt_1h_2020Jan1_2025Mar6.csv',
                        help='Path to the data file')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Path to the experiment directory')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Trading commission as a decimal (default: 0.001 = 0.1%)')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital for the backtest (default: 10000)')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Disable baseline model comparison')
    
    args = parser.parse_args()
    
    run_backtest(
        data_path=args.data,
        experiment_dir=args.experiment,
        commission=args.commission,
        initial_capital=args.capital,
        compare_baseline=not args.no_baseline
    )