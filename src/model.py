import torch
import torch.nn as nn
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the output from the last time step
        x = x[:, -1, :]
        
        # Project to output
        return self.output_layer(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class BaselineModel(nn.Module):
    def __init__(self, window_size=20):
        super(BaselineModel, self).__init__()
        self.window_size = window_size
    
    def forward(self, x):
        # Simple moving average strategy
        # Assuming the first feature is price
        prices = x[:, :, 0]
        sma = torch.mean(prices[:, -self.window_size:], dim=1)
        current_price = prices[:, -1]
        
        # Return 1 if current price > SMA (bullish), -1 otherwise (bearish)
        signal = torch.where(current_price > sma, 
                             torch.tensor(1.0, device=x.device), 
                             torch.tensor(-1.0, device=x.device))
        
        return signal.unsqueeze(1)