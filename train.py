import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import project modules
from src.model import TransformerModel, BaselineModel
from src.process_data import load_data, add_technical_indicators, create_sequences

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs=50,
    device='cpu',
    patience=10,
    experiment_dir=None
):
    """
    Train the model with early stopping
    """
    model.to(device)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the model
            if experiment_dir:
                torch.save(model.state_dict(), os.path.join(experiment_dir, 'model.pt'))
                
                # Save training curve
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, label='Training Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join(experiment_dir, 'loss_curve.png'))
                plt.close()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    return train_losses, val_losses

def main():
    # Configuration
    data_path = 'data/raw/btc_usdt_1h_2020Jan1_2025Mar6.csv'
    processed_path = 'data/processed/btc_usdt_1h_features.csv'
    experiment_name = f'run_{time.strftime("%Y%m%d_%H%M%S")}'
    experiment_dir = os.path.join('experiments', experiment_name)
    
    # Create experiment directory
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Model parameters
    seq_length = 48  # 48 hours
    hidden_dim = 64
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 100
    patience = 15
    
    # Save parameters
    params = {
        'seq_length': seq_length,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'patience': patience
    }
    
    with open(os.path.join(experiment_dir, 'params.txt'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and process data if needed
    if not os.path.exists(processed_path):
        print("Processing raw data...")
        from src.process_data import process_and_save
        df_features = process_and_save(data_path, processed_path, seq_length)
    else:
        print("Loading processed data...")
        df_features = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    
    print(f"Features shape: {df_features.shape}")
    
    # Create sequences
    feature_cols = [col for col in df_features.columns if col != 'returns']
    X, y = create_sequences(df_features, seq_length, 'returns', feature_cols)
    
    print(f"Sequence data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )
    
    # Standardize features (using only training data statistics)
    # First reshape to 2D
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
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = X_train.shape[-1]
    model = TransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Train the model
    print("Training model...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        patience=patience,
        experiment_dir=experiment_dir
    )
    
    # Save the scaler for later use in backtesting
    torch.save(scaler, os.path.join(experiment_dir, 'scaler.pt'))
    
    # Train baseline model for comparison
    print("Training baseline model...")
    baseline_model = BaselineModel(window_size=20)
    
    # Evaluate on test set
    model.load_state_dict(torch.load(os.path.join(experiment_dir, 'model.pt')))
    model.eval()
    
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
    
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    
    # Save test results
    with open(os.path.join(experiment_dir, 'test_results.txt'), 'w') as f:
        f.write(f'Test Loss: {test_loss:.4f}\n')
    
    print(f"Training complete. Results saved to {experiment_dir}")

if __name__ == "__main__":
    main()