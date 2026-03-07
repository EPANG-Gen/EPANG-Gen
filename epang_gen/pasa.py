"""
Bayesian PASA: Prescient Adaptation of Spectral Analysis
"""

import numpy as np


class BayesianPASA:
    """
    Adaptive rank selection based on eigenvalue uncertainties.
    
    Args:
        initial_rank: starting rank (default: 10)
        uncertainty_threshold: threshold for rank increase (default: 0.1)
        max_rank: maximum allowed rank (default: 50)
    """
    
    def __init__(self, initial_rank=10, uncertainty_threshold=0.1, max_rank=50):
        self.rank = initial_rank
        self.threshold = uncertainty_threshold
        self.max_rank = max_rank
        self.history = []

    def update_rank(self, eigenvalues, eigenvalue_uncertainties):
        """
        Update rank based on eigenvalue uncertainties.
        
        Args:
            eigenvalues: array of eigenvalues
            eigenvalue_uncertainties: array of uncertainties
            
        Returns:
            updated rank
        """
        # Check for high uncertainty
        high_uncertainty = eigenvalue_uncertainties > self.threshold
        
        # If any eigenvalue has high uncertainty, increase rank
        if high_uncertainty.any():
            self.rank = min(self.rank + 5, self.max_rank)
        
        # Record history
        self.history.append({
            'rank': self.rank,
            'max_uncertainty': eigenvalue_uncertainties.max(),
            'mean_uncertainty': eigenvalue_uncertainties.mean()
        })
        
        return self.rank

    def get_history(self):
        """Return adaptation history."""
        return self.history
