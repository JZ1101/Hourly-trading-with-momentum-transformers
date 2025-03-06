import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Union
import torch

class Backtester:
    def __init__(
        self, 
        price_data: pd.DataFrame, 
        commission: float = 0.001, 
        initial_capital: float = 10000
    ):
        """
        Backtester for cryptocurrency trading strategies
        
        Args:
            price_data: DataFrame with at least 'close' column and datetime index
            commission: Trading commission as a decimal (e.g., 0.001 = 0.1%)
            initial_capital: Starting capital for the backtest
        """
        self.price_data = price_data
        self.commission = commission
        self.initial_capital = initial_capital
        
    def backtest(
        self, 
        signals: np.ndarray, 
        start_idx: int
    ) -> Dict[str, Any]:
        """
        Run backtest on a set of trading signals
        
        Args:
            signals: Array of signal values (-1 for short, 0 for no position, 1 for long)
            start_idx: Start index in the price data (to align with signals)
            
        Returns:
            Dictionary with backtest results
        """
        # Ensure signals align with price data
        if start_idx + len(signals) > len(self.price_data):
            raise ValueError("Signals extend beyond available price data")
        
        # Extract relevant price data
        prices = self.price_data['close'].iloc[start_idx:start_idx+len(signals)].values
        
        # Initialize backtest variables with safe numeric types
        position = 0  # Current position: -1 (short), 0 (none), 1 (long)
        capital = float(self.initial_capital)
        holdings = 0.0  # Number of units held
        trades = []    # List to track trades
        
        # Tracking metrics
        portfolio_values = [float(capital)]  # Initial portfolio value
        positions = [0]  # Position history
        
        # Fixed position sizing (no compounding)
        position_size = float(self.initial_capital)
        
        # Run backtest
        for i in range(len(signals)):
            try:
                current_price = float(prices[i])
                target_position = int(np.sign(signals[i]))  # Convert to -1, 0, or 1
                
                # Skip invalid prices
                if not np.isfinite(current_price) or current_price <= 0:
                    portfolio_values.append(portfolio_values[-1])
                    positions.append(position)
                    continue
                
                # Check if position change is needed
                if target_position != position:
                    # Close existing position if any
                    if position != 0:
                        # Calculate profit/loss
                        if position == 1:  # Long position
                            # Calculate P&L
                            entry_value = position_size 
                            exit_value = holdings * current_price
                            pnl = exit_value - entry_value
                            capital += pnl - (entry_value + exit_value) * self.commission
                            
                            # Record trade
                            profit_pct = ((current_price / entry_price) - 1) * 100
                        else:  # Short position
                            # Calculate P&L
                            entry_value = position_size
                            exit_value = short_units * current_price
                            pnl = entry_value - exit_value
                            capital += pnl - (entry_value + exit_value) * self.commission
                            
                            # Record trade
                            profit_pct = ((entry_price / current_price) - 1) * 100
                        
                        # Record the trade
                        trades.append({
                            'entry_time': self.price_data.index[start_idx + entry_idx],
                            'exit_time': self.price_data.index[start_idx + i],
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': position,
                            'profit_pct': profit_pct,
                            'holding_period': i - entry_idx
                        })
                        
                        holdings = 0.0
                        short_units = 0.0
                    
                    # Open new position if target is not neutral
                    if target_position != 0:
                        entry_price = current_price
                        entry_idx = i
                        
                        # Fixed position sizing (no compounding)
                        effective_capital = min(capital, position_size)
                        
                        # Calculate position size
                        if target_position == 1:  # Long position
                            holdings = effective_capital * (1 - self.commission) / current_price
                            short_units = 0.0
                        else:  # Short position
                            short_units = effective_capital * (1 - self.commission) / current_price
                            holdings = 0.0
                    
                    # Update position
                    position = target_position
                
                # Calculate current portfolio value
                if position == 0:
                    portfolio_value = capital
                elif position == 1:  # Long
                    # Value of long position
                    market_value = holdings * current_price
                    cost_basis = position_size
                    unrealized_pnl = market_value - cost_basis
                    portfolio_value = capital + unrealized_pnl
                else:  # Short
                    # Value of short position
                    market_value = short_units * current_price
                    cost_basis = position_size
                    unrealized_pnl = cost_basis - market_value
                    portfolio_value = capital + unrealized_pnl
                
                # Safety checks
                if not np.isfinite(portfolio_value) or portfolio_value < 0:
                    portfolio_value = portfolio_values[-1]
                
                # Limit maximum growth per step
                if len(portfolio_values) > 0:
                    prev_value = portfolio_values[-1]
                    max_step = prev_value * 1.1  # Max 10% growth per step
                    portfolio_value = min(portfolio_value, max_step)
                
                # Protect against overflow
                portfolio_value = min(portfolio_value, 1e10)
                
                portfolio_values.append(portfolio_value)
                positions.append(position)
                
            except Exception as e:
                print(f"Error at step {i}: {e}")
                # In case of error, just continue with previous values
                portfolio_values.append(portfolio_values[-1] if portfolio_values else self.initial_capital)
                positions.append(position)
        
        # Calculate performance metrics (with safety checks)
        try:
            final_value = portfolio_values[-1]
            total_return = (final_value / self.initial_capital - 1) * 100
        except:
            final_value = self.initial_capital
            total_return = 0.0
        
        # Calculate safe returns
        safe_values = np.array([v for v in portfolio_values if np.isfinite(v)])
        if len(safe_values) > 1:
            returns = np.diff(safe_values) / safe_values[:-1]
            returns = returns[np.isfinite(returns)]
        else:
            returns = np.array([0.0])
        
        results = {
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': self._calculate_annual_return(safe_values),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'trades': trades,
            'portfolio_values': portfolio_values,
            'positions': positions,
            'returns': returns
        }
        
        return results
    
    def _calculate_annual_return(self, portfolio_values):
        """Calculate annualized return with safety checks"""
        try:
            if len(portfolio_values) < 2:
                return 0.0
            
            first_value = portfolio_values[0]
            last_value = portfolio_values[-1]
            
            if first_value <= 0 or not np.isfinite(first_value) or not np.isfinite(last_value):
                return 0.0
            
            total_return = last_value / first_value
            # Assuming hourly data
            n_years = len(portfolio_values) / (365 * 24)
            
            annual_return = (total_return ** (1 / max(n_years, 1e-6)) - 1) * 100
            # Cap at a reasonable maximum
            return min(annual_return, 100.0) if np.isfinite(annual_return) else 0.0
        except:
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio with safety checks"""
        try:
            if len(returns) < 10:
                return 0.0
            
            clean_returns = returns[np.isfinite(returns)]
            if len(clean_returns) < 10 or np.std(clean_returns) == 0:
                return 0.0
            
            sharpe = (np.mean(clean_returns) / np.std(clean_returns)) * np.sqrt(365 * 24)
            return min(max(sharpe, -5.0), 5.0) if np.isfinite(sharpe) else 0.0
        except:
            return 0.0
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown with safety checks"""
        try:
            max_dd = 0.0
            peak = portfolio_values[0]
            
            for value in portfolio_values:
                if not np.isfinite(value):
                    continue
                
                if value > peak:
                    peak = value
                
                if peak > 0:
                    dd = (peak - value) / peak
                    if np.isfinite(dd):
                        max_dd = max(max_dd, dd)
            
            return min(max_dd * 100, 100.0)  # Cap at 100%
        except:
            return 0.0
    
    def plot_results(self, results):
        """Plot backtest results with improved visualization"""
        plt.figure(figsize=(14, 8))
        
        # Get clean portfolio values for plotting
        portfolio_values = np.array(results['portfolio_values'])
        portfolio_values = np.minimum(portfolio_values, np.percentile(portfolio_values, 99) * 2)
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_values)
        plt.title(f'Portfolio Value Over Time (Final: ${results["final_value"]:.2f})')
        plt.grid(True)
        
        # Plot positions
        plt.subplot(2, 1, 2)
        plt.plot(results['positions'])
        plt.title(f'Position Over Time (-1: Short, 0: None, 1: Long)')
        plt.grid(True)
        plt.tight_layout()
        
        # Print summary metrics
        print(f"Final Value: ${results['final_value']:.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Annual Return: {results['annual_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Number of Trades: {len(results['trades'])}")
        
        # Show winning percentage if there are trades
        if results['trades']:
            winning_trades = sum(1 for trade in results['trades'] if trade['profit_pct'] > 0)
            win_rate = winning_trades / len(results['trades']) * 100
            print(f"Win Rate: {win_rate:.2f}%")

def generate_signals_from_model(
    model: torch.nn.Module, 
    features: np.ndarray, 
    device: str = 'cpu'
) -> np.ndarray:
    """
    Generate trading signals from a PyTorch model
    
    Args:
        model: Trained PyTorch model
        features: Input features [num_samples, seq_length, num_features]
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        Numpy array of trading signals
    """
    model.eval()
    signals = []
    
    with torch.no_grad():
        for i in range(len(features)):
            try:
                # Convert to tensor and add batch dimension
                x = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
                
                # Get model prediction
                output = model(x)
                
                # Convert to signal (-1, 0, or 1)
                signal = output.cpu().numpy()[0][0]
                
                # Safety check for invalid values
                if not np.isfinite(signal):
                    signal = 0.0
                    
                signal_value = np.sign(signal)  # Convert to -1, 0, or 1
                
                signals.append(signal_value)
            except Exception as e:
                # In case of error, use neutral signal
                signals.append(0.0)
    
    return np.array(signals)