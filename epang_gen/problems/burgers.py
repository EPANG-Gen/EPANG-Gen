"""
1D Burgers' equation: u_t + u*u_x - ν*u_xx = 0
"""

import torch
import numpy as np


def burgers_loss(model, x_colloc, t_colloc, x_data, t_data, u_data):
    """
    Compute loss for Burgers' equation.
    
    Args:
        model: neural network
        x_colloc: spatial collocation points
        t_colloc: temporal collocation points
        x_data: data spatial points
        t_data: data temporal points
        u_data: data values
        
    Returns:
        combined PDE and data loss
    """
    x_colloc.requires_grad_(True)
    t_colloc.requires_grad_(True)
    
    xt = torch.cat([x_colloc, t_colloc], dim=1)
    u = model(xt)
    
    # Time derivative
    u_t = torch.autograd.grad(
        u, t_colloc, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True
    )[0]
    
    # Spatial derivatives
    u_x = torch.autograd.grad(
        u, x_colloc, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True
    )[0]
    
    u_xx = torch.autograd.grad(
        u_x, x_colloc, 
        grad_outputs=torch.ones_like(u_x), 
        create_graph=True
    )[0]
    
    nu = 0.01  # viscosity
    f = u_t + u * u_x - nu * u_xx
    loss_pde = torch.mean(f**2)
    
    # Data loss (if available)
    u_data_pred = model(torch.cat([x_data, t_data], dim=1))
    loss_data = torch.mean((u_data_pred - u_data)**2)
    
    return loss_pde + loss_data


def generate_burgers(n_colloc=2000, n_data=100):
    """
    Generate data for Burgers' equation.
    
    Args:
        n_colloc: number of collocation points
        n_data: number of data points
        
    Returns:
        x_colloc, t_colloc, x_data, t_data, u_data
    """
    x_colloc = torch.rand(n_colloc, 1) * 2 - 1
    t_colloc = torch.rand(n_colloc, 1) * 1.0
    
    # Generate synthetic data (placeholder - replace with real data)
    x_data = torch.rand(n_data, 1) * 2 - 1
    t_data = torch.rand(n_data, 1) * 1.0
    u_data = torch.sin(torch.pi * x_data) * torch.exp(-0.01 * t_data)
    
    return x_colloc, t_colloc, x_data, t_data, u_data
