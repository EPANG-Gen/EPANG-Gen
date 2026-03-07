"""
Taylor-Green vortex experiment at Re=100,000.
"""

import os
import json
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from epang_gen import EPANGGen, BayesianPASA, BayesianPINN, set_seed
from epang_gen.benchmarks import taylor_green_loss, generate_taylor_green


def run_experiment(config_path, save_results=True):
    """Run Taylor-Green experiment."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    tg_config = config['taylor_green']
    layers = tg_config['layers']
    epochs_adam = tg_config.get('epochs_adam', 1000)
    epochs_epang = tg_config.get('epochs_epang', 2000)
    re = tg_config['reynolds']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    print("Generating data...")
    data = generate_taylor_green(
        n_colloc=tg_config.get('n_colloc', 10000),
        n_ic=tg_config.get('n_ic', 1000)
    )
    data = [d.to(device) for d in data]
    
    torch.cuda.empty_cache()
    
    # Test Adam
    print("\n--- Testing Adam ---")
    set_seed(42)
    model_adam = BayesianPINN(layers).to(device)
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=config['optimizers']['lr'])
    
    adam_losses = []
    adam_failed = False
    
    for epoch in range(epochs_adam):
        def closure():
            opt_adam.zero_grad()
            loss = taylor_green_loss(model_adam, *data, nu=1.0/re)
            loss.backward()
            return loss.item()
        
        loss = opt_adam.step(closure)
        adam_losses.append(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss:.6f}")
        
        if np.isnan(loss) or np.isinf(loss):
            print(f"!!! ADAM FAILED at epoch {epoch} !!!")
            adam_failed = True
            break
    
    torch.cuda.empty_cache()
    
    # Test EPANG-Gen
    print("\n--- Testing EPANG-Gen ---")
    set_seed(42)
    model_epang = BayesianPINN(layers).to(device)
    pasa = BayesianPASA(
        initial_rank=tg_config.get('initial_rank', 10),
        max_rank=tg_config.get('max_rank', 20)
    )
    opt_epang = EPANGGen(
        model_epang.parameters(),
        lr=config['optimizers']['lr'],
        rank=tg_config.get('initial_rank', 10),
        eigen_update_freq=tg_config.get('eigen_update_freq', 50),
        pasa=pasa
    )
    
    epang_losses = []
    epang_failed = False
    
    for epoch in range(epochs_epang):
        def closure():
            opt_epang.zero_grad()
            loss = taylor_green_loss(model_epang, *data, nu=1.0/re)
            loss.backward()
            return loss.item()
        
        loss = opt_epang.step(closure)
        epang_losses.append(loss)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: loss = {loss:.6f}")
        
        if np.isnan(loss) or np.isinf(loss):
            print(f"!!! EPANG-Gen FAILED at epoch {epoch} !!!")
            epang_failed = True
            break
    
    # Results
    results = {
        'problem': 'taylor_green_3d',
        'reynolds': re,
        'adam_failed': adam_failed,
        'adam_final_loss': float(adam_losses[-1]) if not adam_failed else None,
        'adam_epochs_completed': len(adam_losses),
        'epang_failed': epang_failed,
        'epang_final_loss': float(epang_losses[-1]) if not epang_failed else None,
        'epang_epochs_completed': len(epang_losses)
    }
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"Adam failed: {adam_failed}")
    print(f"Adam final loss: {results['adam_final_loss']:.6f}" if not adam_failed else "Adam failed")
    print(f"EPANG-Gen failed: {epang_failed}")
    print(f"EPANG-Gen final loss: {results['epang_final_loss']:.6f}" if not epang_failed else "EPANG-Gen failed")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(adam_losses, label='Adam', color='red', linewidth=2)
    plt.plot(epang_losses, label='EPANG-Gen', color='blue', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Taylor-Green Vortex (Re={re}): Adam vs EPANG-Gen')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_results:
        # Save results
        results_dir = os.path.join('experiments', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, 'taylor_green_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        plt.savefig(os.path.join(results_dir, 'figures', 'taylor_green.png'), dpi=150)
        print(f"\nResults saved to {results_dir}")
    
    plt.show()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/configs/default.yaml')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results')
    args = parser.parse_args()
    
    run_experiment(args.config, save_results=not args.no_save)
