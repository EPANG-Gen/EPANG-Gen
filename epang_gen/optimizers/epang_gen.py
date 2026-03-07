"""
EPANG-Gen optimizer implementation.
"""

import torch
import numpy as np


class EPANGGen(torch.optim.Optimizer):
    """
    Enhanced Physics-Aware Natural Gradient with Generalization.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for momentum (default: (0.9, 0.999))
        eps: term added for numerical stability (default: 1e-8)
        rank: target rank for eigen-decomposition (default: 10)
        oversampling: oversampling parameter (default: 5)
        eigen_update_freq: frequency of eigen updates (default: 100)
        use_curvature_lr: whether to use curvature-adaptive LR (default: True)
        negative_curvature_threshold: threshold for negative curvature (default: 1e-4)
        smoothing: smoothing factor for preconditioner (default: 0.9)
        pasa: BayesianPASA instance for adaptive rank selection
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 rank=10, oversampling=5, eigen_update_freq=100,
                 use_curvature_lr=True, negative_curvature_threshold=1e-4,
                 smoothing=0.9, pasa=None):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.rank = rank
        self.oversampling = oversampling
        self.eigen_update_freq = eigen_update_freq
        self.use_curvature_lr = use_curvature_lr
        self.negative_curvature_threshold = negative_curvature_threshold
        self.smoothing = smoothing
        self.pasa = pasa
        self._step_count = 0
        self._preconditioner = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._grad_sq = None
        self.power_iters = 1

    def _randomized_eigenspace(self, grad_flat):
        """
        Memory-efficient eigen-decomposition using diagonal approximation.
        
        Args:
            grad_flat: flattened gradient vector
            
        Returns:
            V: approximate eigenvectors
            eigvals: approximate eigenvalues
        """
        d = grad_flat.shape[0]
        effective_rank = min(self.rank, d // 100) if d > 10000 else self.rank
        k = min(effective_rank + self.oversampling, d)

        with torch.no_grad():
            # Generate random matrix
            Omega = torch.randn(d, k, device=grad_flat.device, dtype=grad_flat.dtype)

            # Power iterations with diagonal approximation
            Y = Omega
            for _ in range(self.power_iters):
                if self._grad_sq is not None:
                    self._grad_sq.mul_(0.9).add_(grad_flat**2 * 0.1)
                else:
                    self._grad_sq = grad_flat**2
                H_diag = self._grad_sq + 1e-8
                Y = H_diag.unsqueeze(1) * Y

            # QR decomposition
            Q, _ = torch.linalg.qr(Y)

            # Form T = Q^T H Q
            T = Q.t() @ (H_diag.unsqueeze(1) * Q)

            # Small eigendecomposition
            eigvals, eigvecs_T = torch.linalg.eigh(T)
            idx = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[idx][:effective_rank]
            eigvecs_T = eigvecs_T[:, idx][:, :effective_rank]

            # Approximate eigenvectors of H
            V = Q @ eigvecs_T

        return V, eigvals

    def _update_preconditioner(self, grad_flat):
        """
        Update the preconditioner matrix.
        
        Args:
            grad_flat: flattened gradient vector
        """
        V, eigvals = self._randomized_eigenspace(grad_flat)
        scaling = torch.diag(1.0 / torch.sqrt(eigvals + self.defaults['eps']))
        new_precond = V @ scaling @ V.T
        
        if self._preconditioner is None:
            self._preconditioner = new_precond
        else:
            self._preconditioner = (self.smoothing * self._preconditioner +
                                    (1 - self.smoothing) * new_precond)
        
        self._eigenvalues = eigvals
        self._eigenvectors = V
        
        if self.pasa is not None:
            uncertainties = torch.ones_like(eigvals) * 0.05
            self.rank = self.pasa.update_rank(
                eigvals.cpu().numpy(), 
                uncertainties.cpu().numpy()
            )

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: closure that reevaluates the model and returns the loss
            
        Returns:
            loss: computed loss value
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']

            # Collect gradients
            grad_list = []
            shape_list = []
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_list.append(p.grad.view(-1))
                shape_list.append(p.grad.shape)
            
            if not grad_list:
                continue
            
            grad_flat = torch.cat(grad_list)

            # Update preconditioner
            if self._step_count % self.eigen_update_freq == 1:
                torch.cuda.empty_cache()
                self._update_preconditioner(grad_flat)
                torch.cuda.empty_cache()

            # Apply preconditioner
            if self._preconditioner is not None:
                scaled_g_flat = self._preconditioner @ grad_flat
            else:
                scaled_g_flat = grad_flat

            # Update parameters
            idx = 0
            for p, shp in zip(group['params'], shape_list):
                if p.grad is None:
                    continue
                numel = p.numel()
                scaled_g = scaled_g_flat[idx:idx + numel].view(shp)
                idx += numel

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p)
                    state['second_moment'] = torch.zeros_like(p)

                state['step'] += 1
                state['second_moment'].mul_(beta2).addcmul_(
                    p.grad, p.grad, value=1 - beta2
                )
                state['momentum'].mul_(beta1).add_(scaled_g, alpha=1 - beta1)

                if self.use_curvature_lr and self._eigenvalues is not None:
                    median_eig = self._eigenvalues.median().item()
                    lr_adapted = lr / (np.sqrt(median_eig + group['eps']))
                else:
                    lr_adapted = lr

                p.add_(state['momentum'], alpha=-lr_adapted)

        return loss
