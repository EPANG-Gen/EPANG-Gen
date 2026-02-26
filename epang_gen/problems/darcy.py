"""
2D Darcy flow: -div(k grad u) = 1 with u=0 on boundary circle x^2 + y^2 = 1
"""

import torch
import numpy as np


def darcy_2d_loss(model, x_colloc, y_colloc, x_bc, y_bc, u_bc):
    """
    Compute loss for 2D Darcy flow.
    
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
    
    k_val = 1.0  # permeability
    flux_x = k_val * u_x
    flux_y = k_val * u_y
    
    # Divergence of flux
    div_flux_x = torch.autograd.grad(
        flux_x, x_colloc, 
        grad_outputs=torch.ones_like(flux_x), 
        create_graph=True
    )[0]
    
    div_flux_y = torch.autograd.grad(
        flux_y, y_colloc, 
        grad_outputs=torch.ones_like(flux_y), 
        create_graph=True
    )[0]
    
    div_flux = div_flux_x + div_flux_y
    source = 1.0
    
    loss_pde = torch.mean((div_flux + source)**2)
    
    # Boundary conditions
    xy_bc = torch.cat([x_bc, y_bc], dim=1)
    u_bc_pred = model(xy_bc)
    loss_bc = torch.mean((u_bc_pred - u_bc)**2)
    
    return loss_pde + loss_bc


def generate_darcy_2d(n_colloc=5000, n_bc=400):
    """
    Generate data for 2D Darcy flow.
    
    Args:
        n_colloc: number of collocation points
        n_bc: number of boundary points
        
    Returns:
        x_colloc, y_colloc, x_bc, y_bc, u_bc
    """
    x_colloc = torch.rand(n_colloc, 1) * 2 - 1
    y_colloc = torch.rand(n_colloc, 1) * 2 - 1
    
    # Boundary: circle x^2 + y^2 = 1
    theta = torch.rand(n_bc, 1) * 2 * np.pi
    x_bc = torch.cos(theta)
    y_bc = torch.sin(theta)
    u_bc = torch.zeros(n_bc, 1)
    
    return x_colloc, y_colloc, x_bc, y_bc, u_bc
