"""
Bayesian R‑LayerNorm implementation.
"""

import torch
import torch.nn as nn


class BayesianRLayerNorm(nn.Module):
    """
    Bayesian R‑LayerNorm with uncertainty quantification.
    
    Args:
        normalized_shape: input shape
        prior_scale: scale of prior distribution (default: 1.0)
    """
    
    def __init__(self, normalized_shape, prior_scale=1.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.prior_scale = prior_scale
        
        # Scale parameter (gamma)
        self.gamma_mu = nn.Parameter(torch.ones(normalized_shape))
        self.gamma_logvar = nn.Parameter(torch.zeros(normalized_shape))
        
        # Shift parameter (beta)
        self.beta_mu = nn.Parameter(torch.zeros(normalized_shape))
        self.beta_logvar = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x, return_uncertainty=False):
        """
        Forward pass with optional uncertainty estimation.
        
        Args:
            x: input tensor
            return_uncertainty: whether to return uncertainty estimates
            
        Returns:
            normalized tensor and optionally uncertainty estimates
        """
        # Sample parameters during training
        if self.training:
            gamma = self.gamma_mu + torch.exp(0.5 * self.gamma_logvar) * \
                    torch.randn_like(self.gamma_mu)
            beta = self.beta_mu + torch.exp(0.5 * self.beta_logvar) * \
                   torch.randn_like(self.beta_mu)
        else:
            gamma = self.gamma_mu
            beta = self.beta_mu

        # Normalize
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        normalized = gamma * (x - mean) / std + beta

        if return_uncertainty:
            # Compute uncertainty using variance of parameters
            gamma_std = torch.exp(0.5 * self.gamma_logvar)
            beta_std = torch.exp(0.5 * self.beta_logvar)
            
            # Propagate uncertainty through normalization
            uncertainty = torch.abs(gamma_std * (x - mean) / std) + beta_std
            return normalized, uncertainty

        return normalized
