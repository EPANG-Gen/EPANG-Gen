"""
Utility functions for metrics, visualization, and statistics.
"""

from .metrics import compute_relative_error, compute_pde_residual
from .visualization import plot_convergence, plot_final_comparison, plot_boxplot
from .statistics import compute_statistics, t_test, wilcoxon_test

__all__ = [
    "compute_relative_error",
    "compute_pde_residual",
    "plot_convergence",
    "plot_final_comparison",
    "plot_boxplot",
    "compute_statistics",
    "t_test",
    "wilcoxon_test",
]
