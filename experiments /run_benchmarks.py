"""
Run all benchmark experiments.
"""

import os
import json
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from epang_gen import EPANGGen, BayesianPASA, BayesianPINN, set_seed
from epang_gen.optimizer import ManualADOPT


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_problem(problem_name, optimizer_name, seed, config):
    """Train a single problem with given optimizer."""
    set_seed(seed)
    
    # Import problem-specific functions
    if problem_name == 'poisson_1d':
        from epang_gen.benchmarks import poisson_1d_loss, generate_poisson_1d
        layers = config['poisson_1d']['layers']
        data_fn = generate_poisson_1d
        loss_fn = poisson_1d_loss
        epochs = config['poisson_1d']['epochs']
    elif problem_name == 'burgers':
        from epang_gen.benchmarks import burgers_loss, generate_burgers
        layers = config['burgers']['layers']
        data_fn = generate_burgers
        loss_fn = burgers_loss
        epochs = config['burgers']['epochs']
    elif problem_name == 'darcy_2d':
        from epang_gen.benchmarks import darcy_2d_loss, generate_darcy_2d
        layers = config['darcy_2d']['layers']
        data_fn = generate_darcy_2d
        loss_fn = darcy_2d_loss
        epochs = config['darcy_2d']['epochs']
    elif problem_name == 'helmholtz_2d':
        from epang_gen.benchmarks import helmholtz_2d_loss, generate_helmholtz_2d
        layers = config['helmholtz_2d']['layers']
        data_fn = generate_helmholtz_2d
        loss_fn = helmholtz_2d_loss
        epochs = config['helmholtz_2d']['epochs']
    else:
        raise ValueError(f"Unknown problem: {problem_name}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BayesianPINN(layers).to(device)
    model.train()
    
    # Generate data
    data_args = data_fn()
    data_args = [arg.to(device) if torch.is_tensor(arg) else arg for arg in data_args]
    
    # Create optimizer
    if optimizer_name == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=config['optimizers']['lr'])
    elif optimizer_name == 'ADOPT':
        opt = ManualADOPT(model.parameters(), lr=config['optimizers']['lr'])
    elif optimizer_name == 'EPANG-Gen':
        pasa = BayesianPASA(initial_rank=config['epang']['initial_rank'])
        opt = EPANGGen(
            model.parameters(),
            lr=config['optimizers']['lr'],
            rank=config['epang']['initial_rank'],
            eigen_update_freq=config['epang']['eigen_update_freq'],
            pasa=pasa
        )
    elif optimizer_name == 'AdamW':
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=config['optimizers']['lr'],
            weight_decay=config['adamw']['weight_decay']
        )
    elif optimizer_name == 'EPANG-Gen-light':
        opt = EPANGGen(
            model.parameters(),
            lr=config['optimizers']['lr'],
            rank=config['epang']['initial_rank'],
            eigen_update_freq=10000,
            use_curvature_lr=False
        )
    elif optimizer_name == 'L-BFGS':
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=config['optimizers']['lr'],
            history_size=config['lbfgs']['history_size']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training loop
    loss_history = []
    for epoch in tqdm(range(epochs), desc=f"{problem_name}-{optimizer_name}-seed{seed}"):
        def closure():
            opt.zero_grad()
            loss = loss_fn(model, *data_args)
            loss.backward()
            return loss.item()
        loss = opt.step(closure)
        loss_history.append(loss)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss = {loss:.6f}")
    
    return {
        'problem': problem_name,
        'optimizer': optimizer_name,
        'seed': seed,
        'loss_history': loss_history,
        'final_loss': loss_history[-1]
    }


def main(args):
    """Main experiment runner."""
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    results_dir = os.path.join('experiments', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    optimizers = config['experiment']['optimizers']
    problems = config['experiment']['problems']
    seeds = config['experiment']['seeds']
    
    all_results = []
    
    # Run experiments
    for opt_name in optimizers:
        for prob_name in problems:
            for seed in seeds:
                print(f"\n=== Running {opt_name} on {prob_name}, seed {seed} ===")
                result = train_problem(prob_name, opt_name, seed, config)
                all_results.append(result)
                
                # Save individual result
                filename = f"results_{opt_name}_{prob_name}_seed{seed}.json"
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'w') as f:
                    json.dump(result, f, indent=2)
    
    # Save all results
    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n=== All experiments complete ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/configs/default.yaml')
    args = parser.parse_args()
    main(args)
