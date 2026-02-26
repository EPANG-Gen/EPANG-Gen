"""
EPANG-Gen: Enhanced Physics-Aware Natural Gradient with Generalization
=======================================================================

A novel optimizer combining eigenvalue-based preconditioning,
Bayesian normalization, and physics-aware constraints for scientific
machine learning and Physics-Informed Neural Networks (PINNs).
"""

__version__ = "1.0.0"

from .optimizers import EPANGGen, ManualADOPT
from .models import BayesianPINN, BayesianRLayerNorm, BayesianPASA
from .problems import (
    poisson_1d_loss,
    burgers_loss,
    darcy_2d_loss,
    helmholtz_2d_loss,
)

__all__ = [
    # Optimizers
    "EPANGGen",
    "ManualADOPT",
    # Models
    "BayesianPINN",
    "BayesianRLayerNorm",
    "BayesianPASA",
    # Problems
    "poisson_1d_loss",
    "burgers_loss",
    "darcy_2d_loss",
    "helmholtz_2d_loss",
]
