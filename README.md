# EPANG-Gen: Enhanced Physics-Aware Natural Gradient with Generalization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-JCP_2026-brightgreen.svg)](paper/)

A novel curvature-aware optimizer with built-in uncertainty quantification for scientific machine learning and physics-informed neural networks (PINNs).

## 📄 Overview

EPANG-Gen is the first optimizer specifically designed for geometric and physical AI that combines:

- **Memory-efficient eigen-preconditioning** - Randomized eigenspace estimation with O(dk) memory
- **Bayesian uncertainty quantification** - Per-activation uncertainty via Bayesian R‑LayerNorm
- **Adaptive rank selection** - PASA dynamically adjusts to problem difficulty

## 🔬 Key Results

|   Benchmark                 | Adam       | EPANG-Gen  | ADOPT          |
| --------------------------- |------------|----------- |----------------|
| Poisson 1D                  | 10.71      | 45.39      | 8.91 (25% NaN) |
| Burgers                     | 0.15       | 0.51       | 0.33 (25% NaN) |
| Darcy 2D                    | 0.29       | 1.84       | 4.94 (25% NaN) |
| Helmholtz 2D                | 20.14      | 1880.82    | NaN (66% NaN)  |
| **Taylor-Green Re=100,000** | **0.0251** | **0.0350** | Failed         |

**Zero NaN failures across 72 runs** - EPANG-Gen is the only optimizer with perfect stability.

## 🚀 Installation

```python
# Clone the repository
git clone https://github.com/EPANG-Gen/EPANG-Gen.git
cd EPANG-Gen

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```
🎯 Quick Start
```python
import torch
from epang_gen import EPANGGen, BayesianPINN

# Define your PINN model
model = BayesianPINN(layers=[2, 100, 100, 100, 1])

# Create EPANG-Gen optimizer
optimizer = EPANGGen(
    model.parameters(),
    lr=1e-3,
    rank=10,
    eigen_update_freq=100
)

# Training loop
for epoch in range(1000):
    loss = compute_pde_loss(model, x, t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Get uncertainty estimates
    uncertainties = model.get_uncertainty(x)
```
📊 Reproducing Results

Run the complete benchmark suite:
```python
# Run all 4 PDE benchmarks
python experiments/run_benchmarks.py

# Run Taylor-Green vortex experiment
python experiments/taylor_green.py --reynolds 100000

# Generate figures and tables
python experiments/generate_figures.py
```
📓 Jupyter Notebooks

Explore interactive examples:

    01_introduction.ipynb - Basic usage

    02_benchmark_results.ipynb - Reproduce paper results

    03_taylor_green_analysis.ipynb - Turbulence analysis

    04_uncertainty_demo.ipynb - Uncertainty visualization

📈 Key Features
1. Memory-Efficient Eigen-Preconditioning
```python
# Automatic rank adaptation
optimizer = EPANGGen(model.parameters(), rank=10, pasa=BayesianPASA())
```
2. Bayesian Uncertainty
```python
# Built-in uncertainty estimation
model = BayesianPINN(layers)
uncertainty = model.forward_with_uncertainty(x)
```
3. Physics-Aware Constraints
```python
# Custom PDE loss with uncertainty weighting
loss = pde_loss(model, x) / (uncertainty + 1e-8)
```
📚 Citation

If you use EPANG-Gen in your research, please cite:
```python
@article{mostafa2026epang,
  title={EPANG-Gen: Enhanced Physics-Aware Natural Gradient with Generalization},
  author={Mostafa, Mohsen},
  journal={Journal of Computational Physics},
  year={2026},
  note={Under review}
}
```
📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
📧 Contact

Mohsen Mostafa - mohsen.mostafa.ai@outlook.com

🙏 Acknowledgments

PDEBench for benchmark datasets

PyTorch team for the deep learning framework

Anonymous reviewers for valuable feedback

    

