import torch
import torch.nn as nn

class DirectionalConfidenceLoss(nn.Module):
    """
    Custom loss function that balances directional accuracy with magnitude precision.
    
    This loss consists of two components:
    1. Directional component: Penalizes incorrect prediction of price movement direction
    2. Magnitude component: Standard MSE loss to ensure prediction magnitude is accurate
    
    Parameters:
    - confidence_weight: Weight for the magnitude component (higher values emphasize magnitude)
    - smooth_sign: Whether to use a smooth approximation of the sign function
    """
    def __init__(self, confidence_weight=0.3, smooth_sign=True):
        super(DirectionalConfidenceLoss, self).__init__()
        self.confidence_weight = confidence_weight
        self.smooth_sign = smooth_sign
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        # For the directional component, we want to compare the signs
        if self.smooth_sign:
            # Smooth approximation of sign function for better gradients
            direction_true = torch.tanh(y_true * 10)
            direction_pred = torch.tanh(y_pred * 10)
            # Calculate directional error (1 - cosine similarity)
            direction_penalty = 0.5 * (1 - torch.mean(direction_true * direction_pred))
        else:
            # Hard sign function approach
            direction_true = torch.sign(y_true)
            direction_pred = torch.sign(y_pred)
            # Calculate percentage of incorrect directions
            direction_penalty = torch.mean((direction_true != direction_pred).float())
        
        # Magnitude component: standard MSE on the prediction
        magnitude_penalty = self.mse(y_pred, y_true)
        
        # Combined loss with weighting
        return direction_penalty + self.confidence_weight * magnitude_penalty

# Adaptive version that focuses more on direction for early training
class AdaptiveDirectionalLoss(nn.Module):
    """
    Adaptive version of DirectionalConfidenceLoss that changes the weight
    of the magnitude component over time.
    
    Parameters:
    - initial_weight: Starting weight for magnitude component
    - final_weight: Final weight for magnitude component
    - warmup_epochs: Number of epochs to linearly transition from initial to final weight
    """
    def __init__(self, initial_weight=0.1, final_weight=0.5, warmup_epochs=10):
        super(AdaptiveDirectionalLoss, self).__init__()
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.directional_loss = DirectionalConfidenceLoss(confidence_weight=initial_weight)
    
    def forward(self, y_pred, y_true):
        return self.directional_loss(y_pred, y_true)
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            # Linear interpolation from initial to final weight
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * (epoch / self.warmup_epochs)
        else:
            weight = self.final_weight
        
        self.directional_loss.confidence_weight = weight
        return weight