"""
Main experiment runner for EPANG-Gen paper.
"""

import os
import json
import time
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from epang_gen.optimizers import EPANGGen, ManualADOPT
from epang_gen.models import BayesianPINN, BayesianPASA
from epang_gen.problems import (
    poisson_1d_loss, generate_poisson_1d,
    burgers_loss, generate_burgers,
    darcy_2d_loss, generate_darcy_2d,
    helmholtz_2d_loss, generate_helmholtz_2d
)


def train_problem(problem_name, optimizer_name, seed, epochs, device, config):
    """Train a single problem with given optimizer and seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model configuration
    if problem_name == 'poisson_1d':
        layers = [1, 50, 50, 1]
        data_fn = generate_poisson_1d
        loss_fn = poisson_1d_loss
    elif problem_name == 'burgers':
        layers = [2, 100, 100, 100, 1]
        data_fn = generate_burgers
        loss_fn = burgers_loss
    elif problem_name == 'darcy_2d':
        layers = [2, 100, 100, 100, 1]
        data_fn = generate_darcy_2d
        loss_fn = darcy_2d_loss
    elif problem_name == 'helmholtz_2d':
        layers = [2, 100, 100, 100, 1]
        data_fn = generate_helmholtz_2d
        loss_fn = helmholtz_2d_loss
    else:
        raise ValueError(f"Unknown problem {problem_name}")

    model = BayesianPINN(layers).to(device)
    model.train()
    data_args = data_fn()
    data_args = [arg.to(device) if torch.is_tensor(arg) else arg for arg in data_args]

    # Optimizer selection
    if optimizer_name == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif optimizer_name == 'ADOPT':
        opt = ManualADOPT(
            model.parameters(), 
            lr=config['lr'], 
            betas=(config['beta1'], config['beta2']), 
            eps=config['eps']
        )
    elif optimizer_name == 'EPANG-Gen':
        pasa = BayesianPASA(initial_rank=config['rank'])
        opt = EPANGGen(
            model.parameters(), 
            lr=config['lr'], 
            rank=config['rank'], 
            eigen_update_freq=config['eigen_update_freq'], 
            pasa=pasa
        )
    elif optimizer_name == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    elif optimizer_name == 'EPANG-Gen-light':
        opt = EPANGGen(
            model.parameters(), 
            lr=config['lr'], 
            rank=config['rank'],
            eigen_update_freq=10000,  # effectively never updates
            use_curvature_lr=False
        )
    elif optimizer_name == 'L-BFGS':
        opt = torch.optim.LBFGS(
            model.parameters(), 
            lr=config['lr'], 
            history_size=config.get('history_size', 10)
        )
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    loss_history = []
    time_start = time.time()
    
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

    time_total = time.time() - time_start
    
    return {
        'problem': problem_name,
        'optimizer': optimizer_name,
        'seed': seed,
        'loss_history': loss_history,
        'final_loss': loss_history[-1],
        'time': time_total
    }


def main():
    parser = argparse.ArgumentParser(description='Run EPANG-Gen experiments')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run experiments
    results = []
    total_runs = (len(config['optimizers']) * 
                  len(config['problems']) * 
                  len(config['seeds']))
    
    print(f"Total runs: {total_runs}")
    
    run_counter = 0
    for opt_name in config['optimizers']:
        for prob_name in config['problems']:
            for seed in config['seeds']:
                run_counter += 1
                print(f"\n=== Run {run_counter}/{total_runs}: {opt_name} on {prob_name}, seed {seed} ===")
                
                result = train_problem(
                    prob_name, opt_name, seed, 
                    config['epochs'], device, config
                )
                results.append(result)
                
                # Save intermediate results
                out_file = os.path.join(
                    args.output, 
                    f"results_{opt_name}_{prob_name}_seed{seed}.json"
                )
                with open(out_file, 'w') as f:
                    json.dump(result, f, indent=2)

    # Save all results
    all_results_file = os.path.join(args.output, 'all_results.json')
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Experiment complete! Results saved to {args.output} ===")


if __name__ == '__main__':
    main()
