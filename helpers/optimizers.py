import torch
from torch.optim import Optimizer

class AdamWBf16Moments(Optimizer):
    """
    This is a custom optimizer modified from the standard torch AdamW, but implements the low-precision optimizer states in Deepseek v3 (https://arxiv.org/html/2412.19437v1):
      - Keeps master (`fp32`) weights for each param.
      - Stores moments (exp_avg, exp_avg_sq) in `bf16`.
      - Applies standard AdamW logic in a mixture of float and bf16.
      - Some of the standard torch AdamW options that are rarely used (like amsgrad) are not implemented here. Don't use this for anything other than "typical training".
    
    IMPORTANT 1/31/25: In tests this does save some memory, but leads to such an increase in training loss that it seems not worth it.
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2):
        """
        Arguments try to match https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
        """
        defaults = dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:
            # Extract hyperparameters
            lr = group['lr']
            betas = group['betas']
            beta1, beta2 = betas[0], betas[1]
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()  # Always float32 for the math
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    # Master param in FP32
                    state['fp32_master'] = p.data.float().clone()
                    # Moments in BF16
                    state['exp_avg'] = torch.zeros_like(p.data, dtype = torch.bfloat16)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype = torch.bfloat16)
                    state['step'] = 0

                fp32_master = state['fp32_master']
                exp_avg_bf16 = state['exp_avg']
                exp_avg_sq_bf16 = state['exp_avg_sq']

                # Weight decay (AdamW)
                if weight_decay != 0:
                    grad.add_(fp32_master, alpha = weight_decay)

                # Convert moments to FP32 for calculations
                exp_avg = exp_avg_bf16.float()
                exp_avg_sq = exp_avg_sq_bf16.float()

                state['step'] += 1
                step_count = state['step']

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step_count
                bias_correction2 = 1 - beta2 ** step_count

                # Compute step
                denom = (exp_avg_sq.div(bias_correction2).sqrt_().add_(eps))
                step_size = lr * (exp_avg.div(bias_correction1).div_(denom))

                # Update master weights
                fp32_master.add_(step_size, alpha = -1.0)

                # Copy master back to BF16 for forward pass
                p.data.copy_(fp32_master)

                # Convert updated moments back to BF16
                exp_avg_bf16.copy_(exp_avg.to(torch.bfloat16))
                exp_avg_sq_bf16.copy_(exp_avg_sq.to(torch.bfloat16))
