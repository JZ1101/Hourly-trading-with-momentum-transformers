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
from tqdm import tqdm

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
    Train the model with early stopping and progress bar
    """
    model.to(device)
    
    # Monitor GPU usage if available
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
        torch.cuda.reset_peak_memory_stats(device)  # Reset peak stats
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Calculate total training time estimate
    total_batches = len(train_loader) + len(val_loader)
    print(f"Training for {num_epochs} epochs with {len(train_loader)} training batches and {len(val_loader)} validation batches per epoch")
    print(f"Total number of batches: {total_batches * num_epochs}")
    start_time = time.time()
    
    # Main training loop with progress bar for epochs
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f"Training (Epoch {epoch+1}/{num_epochs})", leave=False)
        for inputs, targets in train_pbar:
            # Explicitly check tensor device to confirm GPU usage
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item() * inputs.size(0)
            train_loss += batch_loss
            
            # Update progress bar with current batch loss
            train_pbar.set_postfix({"batch_loss": f"{batch_loss/inputs.size(0):.4f}"})
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Progress bar for validation batches
        val_pbar = tqdm(val_loader, desc=f"Validation (Epoch {epoch+1}/{num_epochs})", leave=False)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate epoch time and estimate remaining time
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (epoch + 1) * num_epochs
        remaining_time = max(0, estimated_total_time - elapsed_time)
        
        # Report GPU stats every epoch if using GPU
        if device.type == 'cuda':
            print(f"GPU memory: current={torch.cuda.memory_allocated(device) / 1024**2:.2f} MB, "
                  f"peak={torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
        
        # Print progress with time estimates
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Epoch Time: {epoch_time:.1f}s | '
              f'Remaining: {remaining_time/60:.1f}min')
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the model
            if experiment_dir:
                print(f"Saving improved model with validation loss: {val_loss:.4f}")
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
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    
    # Final GPU stats
    if device.type == 'cuda':
        print(f"Final GPU memory stats:")
        print(f"Current memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Peak memory allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    
    return train_losses, val_losses

def main():
    # gpu 
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    # Configuration
    data_path = 'data/raw/btc_usdt_1h_2020Jan1_2025Mar6.csv'
    processed_path = 'data/processed/btc_usdt_1h_features.csv'
    experiment_name = f'run_{time.strftime("%Y%m%d_%H%M%S")}'
    experiment_dir = os.path.join('experiments', experiment_name)
    
    # Create experiment directory
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Model parameters
    seq_length = 72  # 3 days
    hidden_dim = 128
    num_heads = 8
    num_layers = 4
    dropout = 0.3
    learning_rate = 0.0001
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
    
    # Check CUDA availability in more detail
    if torch.cuda.is_available():
        print(f"CUDA is available with {torch.cuda.device_count()} device(s)")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will be on CPU.")
    
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
    
    # Sample batch to verify device placement
    sample_batch, _ = next(iter(train_loader))
    print(f"Sample batch shape: {sample_batch.shape}")
    
    # Initialize model
    input_dim = X_train.shape[-1]
    model = TransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} total parameters")
    
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