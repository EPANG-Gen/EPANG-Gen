"""
2D Helmholtz equation: u_xx + u_yy + k^2 u = q
with u=0 on boundary of square [-1,1]^2
"""

import torch
import numpy as np


def helmholtz_2d_loss(model, x_colloc, y_colloc, x_bc, y_bc, u_bc):
    """
    Compute loss for 2D Helmholtz equation.
    
    Args:
        model: neural network
        x_colloc: x-coordinates of collocation points
        y_colloc: y-coordinates of collocation points
        x_bc: x-coordinates of boundary points
        y_bc: y-coordinates of boundary points
        u_bc: boundary values
        
    Returns:
        combined PDE and boundary loss
    """
    k = 10.0  # wave number
    
    x_colloc.requires_grad_(True)
    y_colloc.requires_grad_(True)
    
    xy = torch.cat([x_colloc, y_colloc], dim=1)
    u = model(xy)
    
    # First derivatives
    u_x = torch.autograd.grad(
        u, x_colloc, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True
    )[0]
    
    u_y = torch.autograd.grad(
        u, y_colloc, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True
    )[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(
        u_x, x_colloc, 
        grad_outputs=torch.ones_like(u_x), 
        create_graph=True
    )[0]
    
    u_yy = torch.autograd.grad(
        u_y, y_colloc, 
        grad_outputs=torch.ones_like(u_y), 
        create_graph=True
    )[0]
    
    # Source term
    q = torch.sin(k * x_colloc) * torch.sin(k * y_colloc)
    
    # PDE residual
    residual = u_xx + u_yy + k**2 * u - q
    loss_pde = torch.mean(residual**2)
    
    # Boundary conditions
    xy_bc = torch.cat([x_bc, y_bc], dim=1)
    u_bc_pred = model(xy_bc)
    loss_bc = torch.mean((u_bc_pred - u_bc)**2)
    
    return loss_pde + loss_bc


def generate_helmholtz_2d(n_colloc=5000, n_bc=400):
    """
    Generate data for 2D Helmholtz equation.
    
    Args:
        n_colloc: number of collocation points
        n_bc: number of boundary points
        
    Returns:
        x_colloc, y_colloc, x_bc, y_bc, u_bc
    """
    x_colloc = torch.rand(n_colloc, 1) * 2 - 1
    y_colloc = torch.rand(n_colloc, 1) * 2 - 1
    
    # Boundary points on square [-1,1]^2
    x_bc = torch.cat([
        torch.rand(n_bc//2, 1) * 2 - 1,
        torch.ones(n_bc//4, 1),
        -torch.ones(n_bc//4, 1)
    ])
    
    y_bc = torch.cat([
        torch.ones(n_bc//2, 1),
        torch.rand(n_bc//4, 1) * 2 - 1,
        torch.rand(n_bc//4, 1) * 2 - 1
    ])
    
    u_bc = torch.zeros(n_bc, 1)
    
    return x_colloc, y_colloc, x_bc, y_bc, u_bc
