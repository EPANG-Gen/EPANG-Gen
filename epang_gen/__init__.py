"""
EPANG-Gen: Enhanced Physics-Aware Natural Gradient with Generalization

A novel optimizer combining memory-efficient eigen-decomposition with
Bayesian uncertainty quantification for scientific machine learning.
"""

from .optimizer import EPANGGen
from .normalization import BayesianRLayerNorm
from .pasa import BayesianPASA
from .models import BayesianPINN
from .utils import set_seed, count_parameters

__version__ = "1.0.0"
__author__ = "Mohsen Mostafa"

__all__ = [
    "EPANGGen",
    "BayesianRLayerNorm",
    "BayesianPASA",
    "BayesianPINN",
    "set_seed",
    "count_parameters",
]
