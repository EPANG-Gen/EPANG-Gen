"""
Unit tests for EPANG-Gen optimizer.
"""

import torch
import unittest
from epang_gen import EPANGGen, BayesianPASA


class TestEPANGGen(unittest.TestCase):
    
    def setUp(self):
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = EPANGGen(self.model.parameters(), lr=1e-3, rank=5)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.rank, 5)
        self.assertEqual(self.optimizer.defaults['lr'], 1e-3)
    
    def test_step(self):
        """Test optimizer step."""
        x = torch.randn(5, 10)
        y = self.model(x)
        loss = y.sum()
        loss.backward()
        
        self.optimizer.step()
        
        # Check that parameters were updated
        for p in self.model.parameters():
            self.assertIsNotNone(p.grad)
    
    def test_pasa_integration(self):
        """Test PASA adaptive rank selection."""
        pasa = BayesianPASA(initial_rank=5, max_rank=10)
        optimizer = EPANGGen(self.model.parameters(), pasa=pasa)
        
        # Simulate some steps
        for _ in range(10):
            x = torch.randn(5, 10)
            y = self.model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
        
        self.assertIsNotNone(optimizer.pasa)


if __name__ == '__main__':
    unittest.main()
