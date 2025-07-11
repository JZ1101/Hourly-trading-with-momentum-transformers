# Momentum Transformer Trading Project

This project implements a transformer-based model for cryptocurrency trading using momentum and mean reversion signals.

## Project Structure

```
Trading-with-momentum-transformers/
├── data/
│   ├── raw/btc_usdt_1h_2020Jan1_2025Mar6.csv     # Raw hourly data
│   └── processed/              # Features + cleaned data
├── src/
│   ├── model.py                # Transformer + SMA baseline
│   ├── process_data.py         # Clean data, add features (RSI, MACD)
│   └── backtest.py             # Backtest with Sharpe ratio, drawdown
├── train.py                    # Train model → save to experiments/
├── backtest.py                 # Run backtests on trained models
├── experiments/                # Track model versions
│   └── run_yyyymmdd/           # Simple naming
│       ├── model.pt            # Weights
│       └── params.txt          # LR, batch size
|   └── search_test/            # Hyperparameter search results
│       ├── run_yyyymmdd/       # Simple naming
├── README.md                   # Project overview
├── search_test.py              # Hyperparameter search, with pipeline of 1 train + 1 backtest for each pair
├── backtest_testset_only.py    # Backtest model focusing on test set only
└── requirements.txt            # Pinned versions
```

## Getting Started

1. **Install the requirements**
   ```
   pip install -r requirements.txt
   ```

2. **Prepare data**
   Place BTC/USDT hourly data in `data/raw/btc_usdt_1h_2020Jan1_2025Mar6.csv` with columns:
   - timestamp
   - open
   - high
   - low
   - close
   - volume

3. **Train the model**
   ```
   python train.py
   ```
   This will:
   - Process the raw data, adding technical indicators
   - Train both transformer and baseline models
   - Save the models and results to the experiments directory

4. **Backtest the model**
   ```
   python backtest.py --experiment experiments/run_yyyymmdd
   ```
   This will:
   - Load the trained model
   - Generate trading signals
   - Compare against a simple moving average baseline
   - Calculate performance metrics
   - Save results and plots

## Key Features

- **Transformer Architecture**: Uses multi-head attention to capture complex patterns in price data
- **Feature Engineering**: Comprehensive technical indicators for momentum and mean reversion
- **Backtesting Framework**: Realistic performance evaluation with transaction costs
- **Baseline Comparison**: Simple moving average model for benchmarking

## Model Architecture

The momentum transformer model uses a multi-head attention mechanism to learn temporal dependencies in cryptocurrency price data. Key components include:

1. **Input Features**: Technical indicators like RSI, MACD, moving averages, and volatility metrics
2. **Transformer Encoder**: Self-attention layers that can capture both long and short-term patterns
3. **Position Encoding**: Adds information about the sequence position to each time step
4. **Output Layer**: Predicts the directional movement for the next time period

## Command Line Options

### Training
```
python train.py
```

### Backtesting
```
python .\backtest_testset_only.py --experiment .\experiments\run_YYYYMMDD_HHMMSS
```

### Training and backtesting pipeline
```
# Default random search with 30 iterations (basic parameter grid)
python hyperparameter_search.py

# Grid search with default parameter grid (exhaustive)
python hyperparameter_search.py --grid-search

# Random search with aggressive parameter grid
python hyperparameter_search.py --aggressive

# Random search with very aggressive parameter grid
python hyperparameter_search.py --very-aggressive

# Random search with research-level (largest) parameter grid
python hyperparameter_search.py --research

# Grid search with aggressive parameter grid (warning: many combinations)
python hyperparameter_search.py --aggressive --grid-search

# Backtest all trained models (recommended for thorough analysis)
python hyperparameter_search.py --very-aggressive --backtest-all

# Specify number of top models to backtest
python hyperparameter_search.py --aggressive --top-models 10

# hyperparameter search with custom parameters
python search_test.py --data ./data/raw/ETH_usdt_1h_2020JAN01_2025MAR14.csv --aggressive --n-iter 60 --output experiments/ETH_search_new_loss --backtest-all 
python search_test.py --data ./data/raw/BTC_usdt_1h_2020JAN01_2025MAR14.csv --aggressive --n-iter 60 --output experiments/BTC_search_new_loss --backtest-all 
python search_test.py --data ./data/raw/SOL_usdt_1h_2020AUG11_2025MAR14.csv --aggressive --n-iter 60 --output experiments/SOL_search_new_loss --backtest-all 
```



Back testing Options:
- `--experiment`: Path to the experiment directory (required)
- `--data`: Path to the data file (default: data/raw/btc_usdt_1h_2020Jan1_2025Mar6.csv)
- `--commission`: Trading commission as a decimal (default: 0.001 = 0.1%)
- `--capital`: Initial capital for the backtest (default: 10000)
- `--no-baseline`: Disable baseline model comparison

How to Run with Synthetic Data
You can now test your transformer model on a synthetic sine wave by running:
python backtest_testset_only.py --experiment experiments/your_model_directory --synthetic
By default, this will use a sine wave with 20 periods, 100 points per period, and 0.1 noise level.
Customization Options
You can customize the synthetic data with these parameters:
python backtest_testset_only.py --experiment experiments/your_model_directory \
                                --synthetic \
                                --wave-type sine \
                                --periods 20 \
                                --points-per-period 100 \
                                --noise 0.1
Available options:

--wave-type: Choose from sine, cosine, or both (combined waves)
--periods: Number of complete wave cycles
--points-per-period: Data points per wave cycle
--noise: Amount of random noise to add (0.0 for perfect waves)

## Future Improvements

- Add more advanced features like order book data and sentiment analysis
- Implement hyperparameter optimization
- Explore ensemble approaches combining transformer with other models
- Add real-time trading capabilities