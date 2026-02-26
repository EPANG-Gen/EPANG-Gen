"""
1D Poisson equation: -u''(x) = f(x) with u(-1)=u(1)=0
Exact solution: u(x) = sin(πx)
"""

import torch
import numpy as np


def poisson_1d_loss(model, x_colloc, x_bc, u_bc):
    """
    Compute loss for 1D Poisson equation.
    
    Args:
        model: neural network
        x_colloc: collocation points
        x_bc: boundary points
        u_bc: boundary values
        
    Returns:
        combined PDE and boundary loss
    """
    x_colloc.requires_grad_(True)
    u = model(x_colloc)
    
    # First derivative
    u_x = torch.autograd.grad(
        u, x_colloc, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True
    )[0]
    
    # Second derivative
    u_xx = torch.autograd.grad(
        u_x, x_colloc, 
        grad_outputs=torch.ones_like(u_x), 
        create_graph=True
    )[0]
    
    # Source term
    f = torch.pi**2 * torch.sin(torch.pi * x_colloc)
    
    # PDE residual: -u'' = f
    loss_pde = torch.mean((u_xx + f)**2)
    
    # Boundary conditions
    u_bc_pred = model(x_bc)
    loss_bc = torch.mean((u_bc_pred - u_bc)**2)
    
    return loss_pde + loss_bc


def generate_poisson_1d(n_colloc=1000, n_bc=2):
    """
    Generate data for 1D Poisson equation.
    
    Args:
        n_colloc: number of collocation points
        n_bc: number of boundary points
        
    Returns:
        x_colloc, x_bc, u_bc
    """
    x_colloc = torch.rand(n_colloc, 1) * 2 - 1  # [-1, 1]
    x_bc = torch.tensor([[-1.0], [1.0]])
    u_bc = torch.tensor([[0.0], [0.0]])
    
    return x_colloc, x_bc, u_bc
