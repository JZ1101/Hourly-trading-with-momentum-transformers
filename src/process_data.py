import pandas as pd
import numpy as np
import os
from typing import Tuple, List

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and validate the cryptocurrency data
    """
    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Sort by timestamp
    df = df.sort_index()
    
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as features
    """
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # 1. Price-based features
    # Calculate returns
    df_features['returns'] = df_features['close'].pct_change()
    
    # Log returns
    df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
    
    # Volatility (rolling standard deviation of returns)
    df_features['volatility_14'] = df_features['returns'].rolling(window=14).std()
    
    # 2. Moving averages
    for window in [7, 14, 21, 50]:
        df_features[f'sma_{window}'] = df_features['close'].rolling(window=window).mean()
        # Moving average convergence/divergence
        if window in [7, 21]:
            df_features[f'macd_{window}'] = df_features['close'].rolling(window=window).mean() - \
                                        df_features['close'].rolling(window=window*2).mean()
    
    # 3. Momentum indicators
    # Relative Strength Index (RSI)
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df_features['rsi_14'] = calculate_rsi(df_features['close'], 14)
    
    # 4. Volume-based features
    # Volume moving average
    df_features['volume_sma_7'] = df_features['volume'].rolling(window=7).mean()
    
    # On-balance volume (OBV)
    df_features['obv'] = (np.sign(df_features['close'].diff()) * df_features['volume']).fillna(0).cumsum()
    
    # 5. Normalized prices
    # Min-max scaling over rolling window
    for window in [14, 30]:
        rolling_min = df_features['close'].rolling(window=window).min()
        rolling_max = df_features['close'].rolling(window=window).max()
        df_features[f'norm_price_{window}'] = (df_features['close'] - rolling_min) / (rolling_max - rolling_min)
    
    # Drop NaN values resulting from the calculations
    df_features = df_features.dropna()
    
    return df_features

def create_sequences(
    df: pd.DataFrame, 
    seq_length: int, 
    target_col: str = 'returns', 
    feature_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for the transformer model
    
    Args:
        df: DataFrame with features
        seq_length: Length of sequences to create
        target_col: Column to use as target (shifted by 1 to predict next value)
        feature_cols: Columns to use as features (if None, use all except target)
        
    Returns:
        X: Feature sequences [num_samples, seq_length, num_features]
        y: Target values [num_samples, 1]
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    # Create target variable (next period's return)
    target = df[target_col].shift(-1)
    
    # Convert to numpy arrays
    data = df[feature_cols].values
    target = target.values
    
    X, y = [], []
    
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
    
    return np.array(X), np.array(y).reshape(-1, 1)

def process_and_save(
    raw_data_path: str, 
    processed_data_path: str, 
    seq_length: int = 60
) -> None:
    """
    Process raw data and save features for training
    """
    # Load raw data
    df = load_data(raw_data_path)
    
    # Add technical indicators
    df_features = add_technical_indicators(df)
    
    # Save processed features
    df_features.to_csv(processed_data_path)
    
    print(f"Processed data saved to {processed_data_path}")
    print(f"Data shape: {df_features.shape}")
    print(f"Features: {df_features.columns.tolist()}")
    
    return df_features
def split_dataset(X, y, test_size=None, val_size=None, shuffle=None, random_state=None):
    """
    Split dataset into train, validation, and test sets in a consistent way.
    Uses config default values if not specified.
    
    Args:
        X: Features array with shape [samples, sequence_length, features]
        y: Target array with shape [samples]
        test_size: Fraction of data to use for testing (default from config)
        val_size: Fraction of remaining data to use for validation (default from config)
        shuffle: Whether to shuffle data before splitting (default from config)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split
    from src.config import DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, SHUFFLE_SPLIT
    
    # Use config values if not specified
    test_size = test_size if test_size is not None else DEFAULT_TEST_SIZE
    val_size = val_size if val_size is not None else DEFAULT_VAL_SIZE
    shuffle = shuffle if shuffle is not None else SHUFFLE_SPLIT
    
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )
    
    # Second split: separate validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, shuffle=shuffle, random_state=random_state
    )
    
    print(f"Data splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Using test_size={test_size}, val_size={val_size}, shuffle={shuffle}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    raw_path = "../data/raw/btc_usdt_1h_2020Jan1_2025Mar6.csv"
    processed_path = "../data/processed/btc_usdt_1h_features.csv"
    
    # Process and save data
    process_and_save(raw_path, processed_path)