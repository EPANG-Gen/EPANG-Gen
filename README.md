# EPANG-Gen: Enhanced Physics-Aware Natural Gradient with Generalization

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/abs/xxxx.xxxxx)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Colab](https://img.shields.io/badge/Colab-Open-yellow)](https://colab.research.google.com/github/yourusername/EPANG-Gen/blob/main/notebooks/02_full_experiment.ipynb)

EPANG-Gen is a novel optimizer that combines **eigenvalue-based preconditioning**, **Bayesian normalization**, and **physics-aware constraints** for training neural networks on scientific machine learning problems, particularly Physics-Informed Neural Networks (PINNs).

## 📊 Key Results

| Optimizer       | Poisson 1D    |    Burgers    | Darcy 2D      |  Helmholtz 2D  |
|-----------------|---------------|---------------|---------------|----------------|
| Adam            | 10.71 ± 1.69  | 0.15 ± 0.04   | 0.28 ± 0.09   | 20.14 ± 18.21  |
| ADOPT           | 8.79 (median) | 0.32 (median) | 0.82 (median) | NaN            |
| **EPANG-Gen**   | 45.39 ± 2.11  | 0.46 ± 0.03   | 2.24 ± 1.41   | 1520 ± 984     |
| AdamW           | 11.28 ± 2.30  | 0.15 ± 0.04   | 0.31 ± 0.08   | 32.30 ± 13.48  |
| EPANG-Gen-light | 51.36 ± 5.04  | 0.88 ± 0.20   | 3.41 ± 1.90   | 217.57 ± 97.25 |
| L-BFGS          | 49.37 ± 0.51  | 0.77 ± 0.26   | 3.82 ± 0.61   | 1793 ± 1886    |

## ✨ Features

- **Memory-efficient eigen-decomposition** - Randomized subspace iteration with diagonal approximation
- **Bayesian R‑LayerNorm** - Uncertainty-aware normalization layers
- **Bayesian PASA** - Adaptive rank selection based on eigenvalue uncertainty
- **6 optimizers** - Adam, AdamW, ADOPT, L-BFGS, EPANG-Gen, EPANG-Gen-light
- **4 PDE benchmarks** - Poisson 1D, Burgers, Darcy 2D, Helmholtz 2D
- **Publication-ready visualizations** - Automatic plots and statistics
- **Colab compatible** - Runs within 5-hour limit on T4 GPU

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/EPANG-Gen.git
cd EPANG-Gen

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage
```python
import torch
from epang_gen.optimizers import EPANGGen
from epang_gen.models import BayesianPINN
from epang_gen.problems import poisson_1d_loss

# Create model and optimizer
model = BayesianPINN(layers=[1, 50, 50, 1])
optimizer = EPANGGen(model.parameters(), lr=1e-3, rank=10)

# Generate data
x_colloc = torch.rand(1000, 1) * 2 - 1
x_bc = torch.tensor([[-1.0], [1.0]])
u_bc = torch.tensor([[0.0], [0.0]])

# Training loop
for epoch in range(1000):
    def closure():
        optimizer.zero_grad()
        loss = poisson_1d_loss(model, x_colloc, x_bc, u_bc)
        loss.backward()
        return loss.item()
    
    loss = optimizer.step(closure)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss:.6f}")
```



## 📈 Reproducing Paper Results

Option 1: Run Full Experiment (Recommended)

```python
# Run all 72 configurations (takes ~4-5 hours on T4 GPU)
python -m epang_gen.experiments.run_experiment --config configs/default.yaml
```

Option 2: Use Colab Notebook

Open and run: https://img.shields.io/badge/Colab-Open-yellow
Option 3: Quick Test
bash

# Run quick test (2 minutes)
python examples/compare_optimizers.py --quick

## 📊 Visualization Examples

```python
from epang_gen.utils.visualization import plot_convergence, plot_final_comparison

# Load results
results = load_results('results/experiment_results.json')

# Generate publication-quality figures
plot_final_comparison(results, save_path='figures/final_comparison.png')
plot_convergence(results, save_path='figures/convergence_curves.png')
```

## 📝 Citation

```python
@article{yourname2026epang,
  title={EPANG-Gen: Enhanced Physics-Aware Natural Gradient with Generalization},
  author={Your Name and Co-authors},
  journal={Journal of Machine Learning Research},
  year={2026},
  volume={xx},
  pages={1-24}
}
```
## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

    Thanks to the ADOPT authors for their NeurIPS 2024 paper

    Built with PyTorch and scientific Python ecosystem


---

### **2. `setup.py`**
```python
"""
Setup script for EPANG-Gen package.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="epang-gen",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Enhanced Physics-Aware Natural Gradient with Generalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/EPANG-Gen",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/EPANG-Gen/issues",
        "Documentation": "https://github.com/yourusername/EPANG-Gen/docs",
        "Source Code": "https://github.com/yourusername/EPANG-Gen",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
    packages=find_packages(exclude=["tests", "notebooks", "examples"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=5.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
```
