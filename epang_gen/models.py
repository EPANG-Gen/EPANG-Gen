"""
Bayesian PINN models with uncertainty.
"""

import torch
import torch.nn as nn
from .normalization import BayesianRLayerNorm


class BayesianPINN(nn.Module):
    """
    Physics-Informed Neural Network with Bayesian R‑LayerNorm.
    
    Args:
        layers: list of layer sizes
        prior_scale: scale of prior for Bayesian layers
    """
    
    def __init__(self, layers, prior_scale=1.0):
        super().__init__()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # No norm on output layer
                self.norms.append(BayesianRLayerNorm(layers[i + 1], prior_scale))

    def forward(self, x):
        """Standard forward pass."""
        for i, (linear, norm) in enumerate(zip(self.linears[:-1], self.norms)):
            x = linear(x)
            x = norm(x)
            x = torch.tanh(x)
        x = self.linears[-1](x)
        return x

    def forward_with_uncertainty(self, x, num_samples=10):
        """
        Forward pass with Monte Carlo uncertainty estimation.
        
        Args:
            x: input tensor
            num_samples: number of MC samples
            
        Returns:
            mean prediction and uncertainty
        """
        predictions = []
        
        # Enable training mode for sampling
        training_mode = self.training
        self.train()
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        # Restore original mode
        self.train(training_mode)
        
        # Compute statistics
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std

    def get_uncertainty(self, x):
        """
        Get uncertainty estimates from Bayesian layers.
        
        Args:
            x: input tensor
            
        Returns:
            list of uncertainty maps per layer
        """
        uncertainties = []
        h = x
        
        for i, (linear, norm) in enumerate(zip(self.linears[:-1], self.norms)):
            h = linear(h)
            h, unc = norm(h, return_uncertainty=True)
            h = torch.tanh(h)
            uncertainties.append(unc)
        
        return uncertainties
