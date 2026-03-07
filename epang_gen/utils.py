"""
Utility functions for EPANG-Gen.
"""

import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: integer seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total parameter count
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_condition_number(grad_flat, eps=1e-8):
    """
    Estimate condition number from gradient information.
    
    Args:
        grad_flat: flattened gradient
        eps: small constant for numerical stability
        
    Returns:
        estimated condition number
    """
    # Use gradient variance as proxy for curvature
    grad_sq = grad_flat**2
    max_curv = grad_sq.max().item()
    min_curv = grad_sq.min().item() + eps
    return max_curv / min_curv


def to_device(data, device):
    """
    Move data to specified device.
    
    Args:
        data: tensor or list of tensors
        device: target device
        
    Returns:
        data on target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data
