"""
Base Physics-Informed Neural Network with Bayesian layers.
"""

import torch
import torch.nn as nn
from .bayesian_layers import BayesianRLayerNorm


class BayesianPINN(nn.Module):
    """
    Physics-Informed Neural Network with Bayesian layer normalization.
    
    Args:
        layers: list of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        prior_scale: scale of prior distribution for Bayesian layers (default: 1.0)
    """
    def __init__(self, layers, prior_scale=1.0):
        super().__init__()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
            # Add Bayesian norm after hidden layers (not after output)
            if i < len(layers) - 2:
                self.norms.append(BayesianRLayerNorm(layers[i+1], prior_scale))
                
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: input tensor
            
        Returns:
            network output
        """
        for i, (linear, norm) in enumerate(zip(self.linears[:-1], self.norms)):
            x = linear(x)
            x = norm(x)
            x = torch.tanh(x)
            
        # Output layer (no activation, no normalization)
        x = self.linears[-1](x)
        return x
