import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid, train_test_split, ParameterSampler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from src.process_data import split_dataset
# Import your existing functions and modules
# Assuming train.py and backtest_testset_only.py are in the same directory
from src.model import TransformerModel, BaselineModel
from src.process_data import load_data, add_technical_indicators, create_sequences
from train import train_model  # Your existing train_model function
from backtest_testset_only import run_backtest_on_test_set
# Import the defaults from config
from src.config import DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE
def grid_search_with_backtest(
    data_path,
    param_grid,
    test_size=DEFAULT_TEST_SIZE,
    val_size=DEFAULT_VAL_SIZE,
    base_experiment_dir='experiments/search_test',
    max_epochs=100,
    backtest_params={
        'commission': 0.001, 
        'initial_capital': 10000,
        'threshold': 0.0001
    },
    force_cpu=False,
    num_top_models=3,  # Number of top models to backtest
    n_iter=30,         # Number of random combinations to try
    use_random_search=True,  # Whether to use random search instead of grid search
    backtest_all=False,  # Whether to backtest all models
    immediate_backtest=False,  # New parameter: Whether to backtest immediately after training
    random_seed=42     # Random seed for reproducibility
):
    """
    Perform parameter search over hyperparameters with integrated backtesting
    
    Args:
        data_path: Path to raw data file
        param_grid: Dictionary with hyperparameter names and lists of values
        test_size: Fraction of data to use for testing
        val_size: Fraction of remaining data to use for validation
        base_experiment_dir: Base directory to save experiments
        max_epochs: Maximum number of epochs for each model
        backtest_params: Dictionary of parameters for backtesting
        force_cpu: Force CPU usage even if CUDA is available
        num_top_models: Number of top models to run backtest on
        n_iter: Number of random combinations to try (for random search)
        use_random_search: Whether to use random search instead of grid search
        backtest_all: Whether to backtest all models
        immediate_backtest: Whether to backtest immediately after training each model
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with search results
    """
    # Set device
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU memory cache: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Create experiment directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_experiment_dir, f'run_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create a search results directory
    search_results_dir = os.path.join(base_experiment_dir, 'search_results')
    os.makedirs(search_results_dir, exist_ok=True)
    
    # Save parameter grid
    with open(os.path.join(experiment_dir, 'param_grid.json'), 'w') as f:
        json.dump(param_grid, f, indent=4)
    
    # Save backtest parameters
    with open(os.path.join(experiment_dir, 'backtest_params.json'), 'w') as f:
        json.dump(backtest_params, f, indent=4)
    
    # Generate parameter combinations
    if use_random_search:
        np.random.seed(random_seed)
        param_combinations = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_seed))
        total_combinations = len(param_combinations)
        print(f"Using RANDOM SEARCH with {total_combinations} randomly sampled combinations")
    else:
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        print(f"Using GRID SEARCH with all {total_combinations} combinations")
    
    # Estimate total training time based on first few runs
    estimate_per_run = None
    
    # Load and process data
    print(f"Loading and processing data from {data_path}")
    # Check if processed data already exists
    processed_path = data_path.replace('raw', 'processed').replace('.csv', '_features.csv')
    
    if os.path.exists(processed_path):
        print(f"Loading pre-processed data from {processed_path}")
        df_features = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    else:
        print("Processing raw data...")
        df = load_data(data_path)
        df_features = add_technical_indicators(df)
        
        # Save processed data for future use
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_features.to_csv(processed_path)
        print(f"Processed data saved to {processed_path}")
    
    print(f"Features shape: {df_features.shape}")
    
    # Initialize results tracking
    results = []
    
    # For immediate backtest mode, prepare backtest summary
    backtest_summary = []
    if immediate_backtest:
        # Save backtest parameters for this search run
        with open(os.path.join(search_results_dir, 'param_grid.json'), 'w') as f:
            json.dump(param_grid, f, indent=4)
        
        with open(os.path.join(search_results_dir, 'backtest_params.json'), 'w') as f:
            json.dump(backtest_params, f, indent=4)
        
        search_file_base = f"search_results_{timestamp}"
    
    # Iterate through parameter combinations - TRAINING PHASE
    for i, params in enumerate(param_combinations):
        run_start_time = time.time()
        print(f"\n[{i+1}/{total_combinations}] Testing parameters: {params}")
        
        # Create run directory
        run_dir = os.path.join(experiment_dir, f'run_{i:03d}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Save current parameters
        with open(os.path.join(run_dir, 'params.txt'), 'w') as f:
            json.dump(params, f, indent=4)
        
        # Extract current sequence length
        seq_length = params['seq_length']
        
        try:
            # Make sure hidden_dim is divisible by num_heads
            if params['hidden_dim'] % params['num_heads'] != 0:
                print(f"Warning: hidden_dim ({params['hidden_dim']}) must be divisible by num_heads ({params['num_heads']})")
                # Adjust hidden_dim to be divisible by num_heads
                params['hidden_dim'] = (params['hidden_dim'] // params['num_heads']) * params['num_heads']
                print(f"Adjusted hidden_dim to {params['hidden_dim']}")
                
                # Update the saved parameters
                with open(os.path.join(run_dir, 'params.txt'), 'w') as f:
                    json.dump(params, f, indent=4)
            
            # Prepare data with current sequence length
            feature_cols = [col for col in df_features.columns if col != 'returns']
            print(f"Creating sequences with length {seq_length}")
            X, y = create_sequences(df_features, seq_length, 'returns', feature_cols)
            
            print(f"Sequence data shape: X={X.shape}, y={y.shape}")
            # Add data split parameters to the run parameters
            params['test_size'] = test_size
            params['val_size'] = val_size
            # Split data using the standardized function
            X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
                X, y, test_size=test_size, val_size=val_size
            )
            
            print(f"Data splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            # Standardize features (using only training data statistics)
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            scaler = StandardScaler()
            X_train_reshaped = scaler.fit_transform(X_train_reshaped)
            X_val_reshaped = scaler.transform(X_val_reshaped)
            X_test_reshaped = scaler.transform(X_test_reshaped)
            
            # Reshape back to 3D
            X_train = X_train_reshaped.reshape(X_train.shape)
            X_val = X_val_reshaped.reshape(X_val.shape)
            X_test = X_test_reshaped.reshape(X_test.shape)
            
            # Create DataLoaders
            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            test_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32)
            )
            
            batch_size = params['batch_size']
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
            
            # Initialize model with current parameters
            input_dim = X_train.shape[-1]
            model = TransformerModel(
                input_dim=input_dim,
                hidden_dim=params['hidden_dim'],
                nhead=params['num_heads'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
            
            # Count model parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model has {total_params:,} total parameters")
            
            # Initialize optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.MSELoss()
            
            # Train model
            print(f"Training model with patience={params['patience']}")
            train_losses, val_losses = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=max_epochs,
                device=device,
                patience=params['patience'],
                experiment_dir=run_dir
            )
            
            # Save the scaler for later use in backtesting
            torch.save(scaler, os.path.join(run_dir, 'scaler.pt'))
            
            # Load best model for evaluation
            model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt')))
            model.eval()
            
            # Evaluate on test set
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item() * inputs.size(0)
            
            test_loss = test_loss / len(test_loader.dataset)
            print(f"Test Loss: {test_loss:.6f}")
            
            # Save test results
            with open(os.path.join(run_dir, 'test_results.txt'), 'w') as f:
                f.write(f'Test Loss: {test_loss:.6f}\n')
                f.write(f'Best Validation Loss: {min(val_losses):.6f}\n')
                f.write(f'Epochs Completed: {len(train_losses)}\n')
            
            # Calculate training time
            train_time = time.time() - run_start_time
            
            # Record training results
            result = {
                'run_id': i,
                'test_loss': test_loss,
                'best_val_loss': min(val_losses),
                'epochs_completed': len(train_losses),
                'training_time': train_time,
                'model_size': total_params,
                'run_dir': run_dir,  # Save the run directory path
                'backtested': False,  # Will be updated if backtest is run
                **params  # Include all hyperparameters
            }
            
            # If immediate backtest is enabled, run backtest now
            if immediate_backtest:
                backtest_start_time = time.time()
                print(f"\nImmediately backtesting model for run {i}...")
                
                try:
                    # Run backtest
                    transformer_results, baseline_results = run_backtest_on_test_set(
                        data_path=data_path,
                        experiment_dir=run_dir,
                        commission=backtest_params['commission'],
                        initial_capital=backtest_params['initial_capital'],
                        compare_baseline=True,
                        threshold=backtest_params['threshold'],
                        force_cpu=force_cpu
                    )
                    
                    # Add backtest results to result dictionary
                    if transformer_results:
                        result['backtested'] = True
                        result['total_return'] = transformer_results['total_return']
                        result['annual_return'] = transformer_results['annual_return']
                        result['sharpe_ratio'] = transformer_results['sharpe_ratio']
                        result['max_drawdown'] = transformer_results['max_drawdown']
                        result['win_rate'] = sum(1 for trade in transformer_results['trades'] if trade['profit_pct'] > 0) / \
                                           max(len(transformer_results['trades']), 1) * 100
                        result['num_trades'] = len(transformer_results['trades'])
                        
                        # Add baseline comparison if available
                        if baseline_results:
                            result['baseline_return'] = baseline_results['total_return']
                            result['baseline_sharpe'] = baseline_results['sharpe_ratio'] 
                            result['baseline_drawdown'] = baseline_results['max_drawdown']
                            result['outperformance'] = transformer_results['total_return'] - baseline_results['total_return']
                        
                        # Add to backtest summary for separate reporting
                        model_result = {
                            'run_id': i,
                            'test_loss': test_loss,
                            'seq_length': params['seq_length'],
                            'hidden_dim': params['hidden_dim'],
                            'num_heads': params['num_heads'],
                            'num_layers': params['num_layers'],
                            'dropout': params['dropout'],
                            'learning_rate': params['learning_rate'],
                            'batch_size': params['batch_size'],
                            'patience': params['patience'],
                            'total_return': transformer_results['total_return'],
                            'annual_return': transformer_results['annual_return'],
                            'sharpe_ratio': transformer_results['sharpe_ratio'],
                            'max_drawdown': transformer_results['max_drawdown'],
                            'win_rate': sum(1 for trade in transformer_results['trades'] if trade['profit_pct'] > 0) / 
                                        max(len(transformer_results['trades']), 1) * 100,
                            'num_trades': len(transformer_results['trades']),
                            'model_path': run_dir,
                            'backtest_timestamp': timestamp
                        }
                        
                        # Add baseline comparison if available
                        if baseline_results:
                            model_result.update({
                                'baseline_return': baseline_results['total_return'],
                                'baseline_sharpe': baseline_results['sharpe_ratio'],
                                'baseline_drawdown': baseline_results['max_drawdown'],
                                'outperformance': transformer_results['total_return'] - baseline_results['total_return']
                            })
                        
                        backtest_summary.append(model_result)
                        
                        # Calculate backtest time
                        backtest_time = time.time() - backtest_start_time
                        result['backtest_time'] = backtest_time
                        
                        print(f"Backtest completed in {backtest_time:.2f}s. Results:")
                        print(f"  Total Return: {transformer_results['total_return']:.2f}%")
                        print(f"  Sharpe Ratio: {transformer_results['sharpe_ratio']:.2f}")
                        print(f"  Max Drawdown: {transformer_results['max_drawdown']:.2f}%")
                        print(f"  Win Rate: {result['win_rate']:.2f}%")
                        print(f"  Number of Trades: {result['num_trades']}")
                    
                except Exception as e:
                    print(f"Error backtesting model {i}: {str(e)}")
                    result['backtest_error'] = str(e)
            
            # Add result to results list
            results.append(result)
            
            # Save all results so far
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(experiment_dir, 'grid_search_results.csv'), index=False)
            
            # Update time estimate for remaining runs
            if i == 0:
                estimate_per_run = time.time() - run_start_time
                estimated_total = estimate_per_run * total_combinations
                print(f"\nEstimated total grid search time: {estimated_total/3600:.1f} hours")
            elif i < 3:
                estimate_per_run = (estimate_per_run * i + (time.time() - run_start_time)) / (i + 1)
                estimated_total = estimate_per_run * total_combinations
                print(f"\nUpdated estimated total grid search time: {estimated_total/3600:.1f} hours")
            
            remaining_combinations = total_combinations - (i + 1)
            estimated_remaining = estimate_per_run * remaining_combinations
            print(f"Estimated remaining time: {estimated_remaining/3600:.1f} hours\n")
            
            # If we're doing immediate backtesting, save the backtest summary after each model
            if immediate_backtest and backtest_summary:
                # Save current backtest summary
                backtest_df = pd.DataFrame(backtest_summary)
                backtest_df.to_csv(os.path.join(search_results_dir, f'{search_file_base}_ongoing.csv'), index=False)
                
                # Also sort and save by total_return for quick reference
                if len(backtest_df) > 1:
                    sorted_df = backtest_df.sort_values('total_return', ascending=False)
                    sorted_df.to_csv(os.path.join(search_results_dir, f'{search_file_base}_by_return_ongoing.csv'), index=False)
                    
                    # Print current best model
                    best_row = sorted_df.iloc[0]
                    print(f"\nCurrent best model (Run {int(best_row['run_id'])}): Return: {best_row['total_return']:.2f}%, "
                          f"Sharpe: {best_row['sharpe_ratio']:.2f}, "
                          f"DrawDown: {best_row['max_drawdown']:.2f}%")
            
        except Exception as e:
            print(f"Error in run {i}: {str(e)}")
            # Add failed run to results with error message
            results.append({
                'run_id': i,
                'error': str(e),
                **params
            })
            continue
    
    # Create final results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by test loss
    if len(results_df) > 0:
        results_df = results_df.sort_values('test_loss')
        
        # Save sorted results
        results_df.to_csv(os.path.join(experiment_dir, 'grid_search_results_sorted.csv'), index=False)
        
        # Log best parameters
        best_params = results_df.iloc[0].to_dict()
        with open(os.path.join(experiment_dir, 'best_params.json'), 'w') as f:
            json.dump({k: v for k, v in best_params.items() if k in param_grid}, f, indent=4)
        
        print(f"\nGrid search complete. Best parameters:")
        for param in param_grid.keys():
            print(f"{param}: {best_params[param]}")
        print(f"Test Loss: {best_params['test_loss']:.6f}")
    else:
        print("No successful runs to report")
        return results_df
    
    # If we've already run backtests immediately, we don't need the separate backtesting phase
    if immediate_backtest:
        # Just finalize the backtest summary
        if backtest_summary:
            backtest_df = pd.DataFrame(backtest_summary)
            
            # Save in search_results directory
            backtest_df.to_csv(os.path.join(search_results_dir, f'{search_file_base}.csv'), index=False)
            
            # Also save sorted by various performance metrics
            metrics = [
                ('total_return', False),  # metric name, ascending flag
                ('sharpe_ratio', False),
                ('max_drawdown', True),
                ('win_rate', False),
                ('test_loss', True)
            ]
            
            for metric, ascending in metrics:
                sorted_df = backtest_df.sort_values(metric, ascending=ascending)
                sorted_df.to_csv(os.path.join(search_results_dir, f'{search_file_base}_by_{metric}.csv'), index=False)
            
            # Create a summary file with key statistics
            with open(os.path.join(search_results_dir, f'{search_file_base}_summary.txt'), 'w') as f:
                f.write(f"Hyperparameter Search Summary ({timestamp})\n")
                f.write(f"===================================================\n\n")
                f.write(f"Total models tested: {len(results)}\n")
                f.write(f"Models backtested: {len(backtest_summary)}\n\n")
                
                f.write(f"Top 10 models by return:\n")
                for i, row in backtest_df.sort_values('total_return', ascending=False).head(10).iterrows():
                    f.write(f"Run {int(row['run_id'])}: Return: {row['total_return']:.2f}%, "
                        f"Sharpe: {row['sharpe_ratio']:.2f}, "
                        f"Drawdown: {row['max_drawdown']:.2f}%, "
                        f"Win Rate: {row['win_rate']:.2f}%\n")
                
                f.write(f"\nTop 10 models by Sharpe ratio:\n")
                for i, row in backtest_df.sort_values('sharpe_ratio', ascending=False).head(10).iterrows():
                    f.write(f"Run {int(row['run_id'])}: Sharpe: {row['sharpe_ratio']:.2f}, "
                        f"Return: {row['total_return']:.2f}%, "
                        f"Drawdown: {row['max_drawdown']:.2f}%, "
                        f"Win Rate: {row['win_rate']:.2f}%\n")
                
                f.write(f"\nTop 10 models by lowest drawdown (with positive returns):\n")
                positive_returns = backtest_df[backtest_df['total_return'] > 0]
                for i, row in positive_returns.sort_values('max_drawdown', ascending=True).head(10).iterrows():
                    f.write(f"Run {int(row['run_id'])}: Drawdown: {row['max_drawdown']:.2f}%, "
                        f"Return: {row['total_return']:.2f}%, "
                        f"Sharpe: {row['sharpe_ratio']:.2f}, "
                        f"Win Rate: {row['win_rate']:.2f}%\n")
            
            # Print summary to console
            print("\nBacktest summary:")
            print(f"Top 10 models by return:")
            top_models = backtest_df.sort_values('total_return', ascending=False).head(10)
            for i, row in top_models.iterrows():
                print(f"Run {int(row['run_id'])}: Return: {row['total_return']:.2f}%, "
                    f"Sharpe: {row['sharpe_ratio']:.2f}, "
                    f"Drawdown: {row['max_drawdown']:.2f}%, "
                    f"Win Rate: {row['win_rate']:.2f}%")
            
            print(f"\nResults saved to: {os.path.join(search_results_dir, search_file_base)}_*.csv")
            
        return results_df
    
    # BACKTESTING PHASE - Run this phase only if we didn't do immediate backtesting
    if len(results) > 0 and not immediate_backtest:
        if backtest_all:
            print(f"\nRunning backtests on ALL {len(results)} models...")
            models_to_backtest = results
        else:
            # Sort results to identify top models to backtest
            sorted_results = sorted(results, key=lambda x: x['test_loss'])
            models_to_backtest = sorted_results[:min(num_top_models, len(sorted_results))]
            print(f"\nRunning backtests on top {len(models_to_backtest)} models...")
        
        # Create a summary dataframe to track backtest results
        backtest_summary = []
        
        # Save parameter grid information in search results
        with open(os.path.join(search_results_dir, 'param_grid.json'), 'w') as f:
            json.dump(param_grid, f, indent=4)
        
        # Save backtest parameters
        with open(os.path.join(search_results_dir, 'backtest_params.json'), 'w') as f:
            json.dump(backtest_params, f, indent=4)
        
        # Create a timestamp for this search run
        search_timestamp = timestamp  # Use the same timestamp as the experiment directory
        search_file_base = f"search_results_{search_timestamp}"
        
        for i, model_info in enumerate(models_to_backtest):
            print(f"\nBacktesting model {i+1}/{len(models_to_backtest)}: Run {model_info['run_id']}")
            print(f"Test Loss: {model_info['test_loss']:.6f}")
            print(f"Parameters: ", end="")
            for key in param_grid.keys():
                print(f"{key}={model_info[key]}", end=", ")
            print("\n")
            
            try:
                # Run backtest
                transformer_results, baseline_results = run_backtest_on_test_set(
                    data_path=data_path,
                    experiment_dir=model_info['run_dir'],
                    commission=backtest_params['commission'],
                    initial_capital=backtest_params['initial_capital'],
                    compare_baseline=True,
                    threshold=backtest_params['threshold'],
                    force_cpu=force_cpu
                )
                
                # Add backtest results to summary
                if transformer_results:
                    model_result = {
                        'run_id': model_info['run_id'],
                        'test_loss': model_info['test_loss'],
                        'seq_length': model_info['seq_length'],
                        'hidden_dim': model_info['hidden_dim'],
                        'num_heads': model_info['num_heads'],
                        'num_layers': model_info['num_layers'],
                        'dropout': model_info['dropout'],
                        'learning_rate': model_info['learning_rate'],
                        'batch_size': model_info['batch_size'],
                        'patience': model_info['patience'],
                        'total_return': transformer_results['total_return'],
                        'annual_return': transformer_results['annual_return'],
                        'sharpe_ratio': transformer_results['sharpe_ratio'],
                        'max_drawdown': transformer_results['max_drawdown'],
                        'win_rate': sum(1 for trade in transformer_results['trades'] if trade['profit_pct'] > 0) / 
                                    max(len(transformer_results['trades']), 1) * 100,
                        'num_trades': len(transformer_results['trades']),
                        'model_path': model_info['run_dir'],
                        'backtest_timestamp': search_timestamp
                    }
                    
                    # Add baseline comparison if available
                    if baseline_results:
                        model_result.update({
                            'baseline_return': baseline_results['total_return'],
                            'baseline_sharpe': baseline_results['sharpe_ratio'],
                            'baseline_drawdown': baseline_results['max_drawdown'],
                            'outperformance': transformer_results['total_return'] - baseline_results['total_return']
                        })
                    
                    backtest_summary.append(model_result)
                
                # Add a flag to the results indicating this model was backtested
                for j in range(len(results)):
                    if results[j]['run_id'] == model_info['run_id']:
                        results[j]['backtested'] = True
                        results[j]['total_return'] = transformer_results['total_return']
                        results[j]['sharpe_ratio'] = transformer_results['sharpe_ratio']
                        results[j]['max_drawdown'] = transformer_results['max_drawdown']
                        break
                
            except Exception as e:
                print(f"Error backtesting model {model_info['run_id']}: {str(e)}")
                continue
        
        # Save backtest summary if we have results
        if backtest_summary:
            backtest_df = pd.DataFrame(backtest_summary)
            
            # Save in search_results directory
            backtest_df.to_csv(os.path.join(search_results_dir, f'{search_file_base}.csv'), index=False)
            
            # Also save sorted by various performance metrics
            metrics = [
                ('total_return', False),  # metric name, ascending flag
                ('sharpe_ratio', False),
                ('max_drawdown', True),
                ('win_rate', False),
                ('test_loss', True)
            ]
            
            for metric, ascending in metrics:
                sorted_df = backtest_df.sort_values(metric, ascending=ascending)
                sorted_df.to_csv(os.path.join(search_results_dir, f'{search_file_base}_by_{metric}.csv'), index=False)
            
            # Create a summary file with key statistics
            with open(os.path.join(search_results_dir, f'{search_file_base}_summary.txt'), 'w') as f:
                f.write(f"Hyperparameter Search Summary ({search_timestamp})\n")
                f.write(f"===================================================\n\n")
                f.write(f"Total models tested: {len(results)}\n")
                f.write(f"Models backtested: {len(backtest_summary)}\n\n")
                
                f.write(f"Top 10 models by return:\n")
                for i, row in backtest_df.sort_values('total_return', ascending=False).head(10).iterrows():
                    f.write(f"Run {int(row['run_id'])}: Return: {row['total_return']:.2f}%, "
                        f"Sharpe: {row['sharpe_ratio']:.2f}, "
                        f"Drawdown: {row['max_drawdown']:.2f}%, "
                        f"Win Rate: {row['win_rate']:.2f}%\n")
                
                f.write(f"\nTop 10 models by Sharpe ratio:\n")
                for i, row in backtest_df.sort_values('sharpe_ratio', ascending=False).head(10).iterrows():
                    f.write(f"Run {int(row['run_id'])}: Sharpe: {row['sharpe_ratio']:.2f}, "
                        f"Return: {row['total_return']:.2f}%, "
                        f"Drawdown: {row['max_drawdown']:.2f}%, "
                        f"Win Rate: {row['win_rate']:.2f}%\n")
                
                f.write(f"\nTop 10 models by lowest drawdown (with positive returns):\n")
                positive_returns = backtest_df[backtest_df['total_return'] > 0]
                for i, row in positive_returns.sort_values('max_drawdown', ascending=True).head(10).iterrows():
                    f.write(f"Run {int(row['run_id'])}: Drawdown: {row['max_drawdown']:.2f}%, "
                        f"Return: {row['total_return']:.2f}%, "
                        f"Sharpe: {row['sharpe_ratio']:.2f}, "
                        f"Win Rate: {row['win_rate']:.2f}%\n")
            
            # Print summary to console
            print("\nBacktest summary:")
            print(f"Top 10 models by return:")
            top_models = backtest_df.sort_values('total_return', ascending=False).head(10)
            for i, row in top_models.iterrows():
                print(f"Run {int(row['run_id'])}: Return: {row['total_return']:.2f}%, "
                    f"Sharpe: {row['sharpe_ratio']:.2f}, "
                    f"Drawdown: {row['max_drawdown']:.2f}%, "
                    f"Win Rate: {row['win_rate']:.2f}%")
            
            print(f"\nResults saved to: {os.path.join(search_results_dir, search_file_base)}_*.csv")
    
    # Update and save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(experiment_dir, 'parameter_search_results_final.csv'), index=False) #parameter_search_results_final.csv
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Run parameter search with integrated backtesting')
    parser.add_argument('--data', type=str, default='data/raw/btc_usdt_1h_2020Jan1_2025Mar6.csv',
                        help='Path to raw data file')
    parser.add_argument('--output', type=str, default='experiments/detailed_search_test',
                        help='Base directory for experiment outputs')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum number of epochs per model')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Trading commission for backtesting (default: 0.001 = 0.1%)')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital for backtesting (default: 10000)')
    parser.add_argument('--threshold', type=float, default=0.0001,
                        help='Threshold for signal generation (default: 0.0001)')
    parser.add_argument('--top-models', type=int, default=3,
                        help='Number of top models to backtest (default: 3)')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    parser.add_argument('--grid-search', action='store_true',
                        help='Use grid search instead of random search')
    parser.add_argument('--n-iter', type=int, default=30,
                        help='Number of random combinations to try (default: 30)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--aggressive', action='store_true',
                        help='Use more aggressive parameter grid')
    parser.add_argument('--very-aggressive', action='store_true',
                        help='Use very aggressive parameter grid (warning: long runtime)')
    parser.add_argument('--research', action='store_true',
                        help='Use research-level parameter grid (very large networks)')
    parser.add_argument('--backtest-all', action='store_true',
                        help='Backtest all models instead of just top performers')
    parser.add_argument('--immediate-backtest', action='store_true',
                        help='Run backtest immediately after training each model')
    parser.add_argument('--test-size', type=float, default=DEFAULT_TEST_SIZE,
                    help=f'Fraction of data to use for testing (default: {DEFAULT_TEST_SIZE})')
    parser.add_argument('--val-size', type=float, default=DEFAULT_VAL_SIZE,
                    help=f'Fraction of remaining data to use for validation (default: {DEFAULT_VAL_SIZE})')
    
    args = parser.parse_args()
    
    # If backtest-all is specified, enable immediate backtest by default
    immediate_backtest = args.immediate_backtest or args.backtest_all
    
    # Define parameter grid based on aggressiveness level
    if args.research:
        # Research-level parameter grid with very large networks
        param_grid = {
            'seq_length': [8, 13, 21, 34, 55, 89, 144, 233, 377, 610],  # Extended Fibonacci numbers
            'hidden_dim': [128, 256, 512, 1024, 2048],  # Much larger hidden dimensions
            'num_heads': [4, 8, 16, 32],  # More attention heads
            'num_layers': [2, 4, 6, 8, 12],  # Deeper networks
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            'learning_rate': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4],  # More learning rates
            'batch_size': [16, 32, 64, 128],
            'patience': [15, 20, 25, 30]  # More patience
        }
        print("Using RESEARCH-LEVEL parameter grid - this will require significant GPU memory and time!")
        
        # Ensure we're using random search for this massive grid
        if not args.grid_search and args.n_iter < 50:
            print("WARNING: Research grid is very large. Increasing random iterations to 50.")
            args.n_iter = 50
            
    elif args.very_aggressive:
        # Very aggressive parameter grid
        param_grid = {
            'seq_length': [8, 13, 21, 34, 55, 89, 144, 233],  # Fibonacci numbers
            'hidden_dim': [64, 128, 256, 512, 768],
            'num_heads': [4, 8, 16, 24],
            'num_layers': [2, 4, 6, 8],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [16, 32, 64, 128],
            'patience': [10, 15, 20, 25]
        }
        print("Using VERY AGGRESSIVE parameter grid!")
        
    elif args.aggressive:
        # Aggressive parameter grid
        param_grid = {
            'seq_length': [5, 8, 13, 21, 34, 89], # Fibonacci numbers
            'hidden_dim': [64, 128, 256, 384],
            'num_heads': [4, 8, 16, 32],
            'num_layers': [ 4, 6, 8, 12],
            'dropout': [0.1, 0.3, 0.5],
            'learning_rate': [1e-4, 5e-4, 1e-3],
            'batch_size': [16, 32, 64],
            'patience': [10, 15, 20]
        }
        print("Using AGGRESSIVE parameter grid")
        
    else:
        # Default parameter grid
        param_grid = {
            'seq_length': [21, 55, 89],  # Fibonacci numbers
            'hidden_dim': [64, 128, 256],
            'num_heads': [4, 8],
            'num_layers': [2, 4],
            'dropout': [0.2, 0.3, 0.4],
            'learning_rate': [1e-4, 5e-4],
            'batch_size': [32, 64],
            'patience': [15]
        }
        print("Using DEFAULT parameter grid")
    
    # Backtest parameters
    backtest_params = {
        'commission': args.commission,
        'initial_capital': args.capital,
        'threshold': args.threshold
    }
    
    # Count total number of parameter combinations
    from itertools import product
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    
    # Determine if we're using random search
    use_random_search = not args.grid_search
    
    if use_random_search:
        # Estimate based on n_iter instead of total combinations
        n_iter = args.n_iter
        estimated_hours = n_iter * 15 / 60  # Assuming 15 minutes per combination
        print(f"Total possible combinations: {total_combinations}")
        print(f"Using RANDOM SEARCH with {n_iter} combinations")
        print(f"Estimated runtime: {estimated_hours:.1f} hours (rough estimate)")
    else:
        # Grid search estimate
        estimated_hours = total_combinations * 10 / 60
        print(f"Using GRID SEARCH with all {total_combinations} combinations")
        print(f"Estimated runtime: {estimated_hours:.1f} hours (rough estimate)")
    
    # If immediate backtest is enabled, inform the user
    if immediate_backtest:
        print("IMMEDIATE BACKTEST mode enabled: Each model will be backtested directly after training")
        # Adjust time estimate
        estimated_hours *= 1.2  # Add 20% for backtest time
        print(f"Adjusted estimated runtime with backtesting: {estimated_hours:.1f} hours")
    
    # Ask for confirmation if runtime is very long
    if (use_random_search and n_iter > 30) or (not use_random_search and total_combinations > 100):
        confirm = input(f"Warning: This search might take {estimated_hours:.1f} hours. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborting.")
            return
    
    # Run parameter search with integrated backtesting
    results = grid_search_with_backtest(
        data_path=args.data,
        param_grid=param_grid,
        base_experiment_dir=args.output,
        max_epochs=args.max_epochs,
        backtest_params=backtest_params,
        force_cpu=args.force_cpu,
        num_top_models=args.top_models,
        n_iter=args.n_iter,
        use_random_search=use_random_search,
        backtest_all=args.backtest_all,
        immediate_backtest=immediate_backtest,  # Pass the immediate_backtest flag
        random_seed=args.seed
    )
    
    # Print summary of best models
    if len(results) > 0:
        print("\nTop 10 models by test loss:")
        for i, row in results.sort_values('test_loss').head(10).iterrows():
            print(f"Run {int(row['run_id'])}, Test Loss: {row['test_loss']:.6f}, "
                  f"Seq Length: {int(row['seq_length'])}, "
                  f"Hidden Dim: {int(row['hidden_dim'])}, "
                  f"Heads: {int(row['num_heads'])}, "
                  f"Layers: {int(row['num_layers'])}")
        
        # If backtesting was done and results are available, show those too
        if 'total_return' in results.columns:
            print("\nTop 10 models by total return:")
            for i, row in results.sort_values('total_return', ascending=False).head(10).iterrows():
                print(f"Run {int(row['run_id'])}, Return: {row['total_return']:.2f}%, "
                      f"Sharpe: {row['sharpe_ratio']:.2f}, "
                      f"DrawDown: {row['max_drawdown']:.2f}%, "
                      f"Test Loss: {row['test_loss']:.6f}")

if __name__ == "__main__":
    main()