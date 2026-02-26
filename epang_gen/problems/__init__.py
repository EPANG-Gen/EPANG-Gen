"""
PDE problem definitions for benchmarking optimizers.
"""

from .poisson import poisson_1d_loss, generate_poisson_1d
from .burgers import burgers_loss, generate_burgers
from .darcy import darcy_2d_loss, generate_darcy_2d
from .helmholtz import helmholtz_2d_loss, generate_helmholtz_2d

__all__ = [
    "poisson_1d_loss",
    "generate_poisson_1d",
    "burgers_loss",
    "generate_burgers",
    "darcy_2d_loss",
    "generate_darcy_2d",
    "helmholtz_2d_loss",
    "generate_helmholtz_2d",
]
