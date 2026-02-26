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
