"""
Unit tests for optimizer implementations.
"""

import pytest
import torch
from epang_gen.optimizers import EPANGGen, ManualADOPT


def test_epang_gen_creation():
    """Test EPANGGen optimizer creation."""
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = EPANGGen(params, lr=0.01, rank=5)
    assert optimizer.rank == 5
    assert optimizer.defaults['lr'] == 0.01


def test_adopt_creation():
    """Test ManualADOPT optimizer creation."""
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = ManualADOPT(params, lr=0.01)
    assert optimizer.defaults['lr'] == 0.01


def test_epang_gen_step():
    """Test EPANGGen optimizer step."""
    x = torch.tensor([5.0], requires_grad=True)
    optimizer = EPANGGen([x], lr=0.01, rank=1)
    
    for _ in range(10):
        def closure():
            optimizer.zero_grad()
            loss = x**2
            loss.backward()
            return loss.item()
        
        loss = optimizer.step(closure)
    
    assert x.item() < 5.0  # Should decrease


def test_adopt_step():
    """Test ManualADOPT optimizer step."""
    x = torch.tensor([5.0], requires_grad=True)
    optimizer = ManualADOPT([x], lr=0.01)
    
    for _ in range(100):
        def closure():
            optimizer.zero_grad()
            loss = x**2
            loss.backward()
            return loss.item()
        
        loss = optimizer.step(closure)
    
    assert abs(x.item()) < 1.0  # Should converge
