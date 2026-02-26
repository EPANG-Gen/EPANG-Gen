"""
Manual ADOPT implementation with gradient clipping.
Based on the NeurIPS 2024 paper "ADOPT: Modified Adam Can Converge with Any β₂".
"""

import torch


class ManualADOPT(torch.optim.Optimizer):
    """
    Manual implementation of ADOPT with gradient clipping.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for momentum and second moment (β1, β2) (default: (0.9, 0.999))
        eps: term added for numerical stability (default: 1e-8)
        clip_value: gradient clipping value (default: 1.0)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, clip_value=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, clip_value=clip_value)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: callable that evaluates the model and returns loss
            
        Returns:
            loss value
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            clip_value = group['clip_value']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_([p], clip_value)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                prev_grad = state['prev_grad']

                # ADOPT core: use previous gradient for second moment
                if step > 1:
                    exp_avg_sq.mul_(beta2).addcmul_(
                        prev_grad, prev_grad, value=1-beta2
                    )
                    denom = exp_avg_sq.sqrt().add_(eps)
                    normalized_grad = grad / denom
                else:
                    normalized_grad = grad

                exp_avg.mul_(beta1).add_(normalized_grad, alpha=1-beta1)
                p.add_(exp_avg, alpha=-lr)
                prev_grad.copy_(grad)

        return loss
