"""
Global configuration for the momentum transformer trading project.
"""
import os

# Data paths
DATA_ROOT = 'data'
RAW_DATA_PATH = os.path.join(DATA_ROOT, 'raw/btc_usdt_1h_2020Jan1_2025Mar6.csv')
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, 'processed')

# Default dataset split parameters
DEFAULT_TEST_SIZE = 0.1    # 10% of data for testing
DEFAULT_VAL_SIZE = 0.15     # 15% of remaining data for validation
SHUFFLE_SPLIT = False      # Don't shuffle time series data

# Default model parameters
DEFAULT_SEQ_LENGTH = 72    # 3 days (hourly data)
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 4
DEFAULT_DROPOUT = 0.3
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 100
DEFAULT_PATIENCE = 15

# Default backtest parameters
DEFAULT_COMMISSION = 0.001         # 0.1%
DEFAULT_INITIAL_CAPITAL = 10000    # $10,000
DEFAULT_SIGNAL_THRESHOLD = 0.0001  # Threshold for signal generation

# Experiments
EXPERIMENT_DIR = 'experiments'