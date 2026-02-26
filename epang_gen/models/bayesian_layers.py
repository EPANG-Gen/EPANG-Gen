"""
Bayesian normalization layers with uncertainty quantification.
"""

import torch
import torch.nn as nn


class BayesianRLayerNorm(nn.Module):
    """
    Bayesian R‑LayerNorm with learnable uncertainty.
    
    Args:
        normalized_shape: input shape (from expected input of size)
        prior_scale: scale of prior distribution (default: 1.0)
    """
    def __init__(self, normalized_shape, prior_scale=1.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.gamma_mu = nn.Parameter(torch.ones(normalized_shape))
        self.gamma_logvar = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_mu = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_logvar = nn.Parameter(torch.zeros(normalized_shape))
        self.prior_scale = prior_scale

    def forward(self, x):
        """
        Forward pass with Bayesian sampling during training.
        
        Args:
            x: input tensor
            
        Returns:
            normalized tensor
        """
        if self.training:
            # Sample from posterior during training
            gamma = self.gamma_mu + torch.exp(0.5 * self.gamma_logvar) * torch.randn_like(self.gamma_mu)
            beta = self.beta_mu + torch.exp(0.5 * self.beta_logvar) * torch.randn_like(self.beta_mu)
        else:
            # Use mean during evaluation
            gamma = self.gamma_mu
            beta = self.beta_mu
            
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        return gamma * (x - mean) / std + beta


class BayesianPASA:
    """
    Prescient Adaptation of Spectral Analysis with uncertainty.
    Dynamically adjusts eigen-rank based on eigenvalue uncertainties.
    """
    def __init__(self, initial_rank=10, uncertainty_threshold=0.1, max_rank=50):
        self.rank = initial_rank
        self.threshold = uncertainty_threshold
        self.max_rank = max_rank

    def update_rank(self, eigenvalues, eigenvalue_uncertainties):
        """
        Update rank based on eigenvalue uncertainties.
        
        Args:
            eigenvalues: array of eigenvalues
            eigenvalue_uncertainties: array of uncertainties
            
        Returns:
            updated rank
        """
        high_uncertainty = eigenvalue_uncertainties > self.threshold
        if high_uncertainty.any():
            self.rank = min(self.rank + 5, self.max_rank)
        return self.rank
