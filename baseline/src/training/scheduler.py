"""
LR Scheduler
==============
- cosine_warmup: Warmup + Cosine Decay (single cycle)
- cosine_warm_restart: Warmup + Cosine with Warm Restarts (SGDR)
"""

import math
import torch


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.05,
                    min_lr: float = 1e-6, mode: str = "cosine_warmup",
                    restart_period: int = 0):
    """
    Args:
        optimizer: optimizer instance
        total_steps: total training steps (for THIS phase)
        warmup_ratio: fraction of steps for warmup
        min_lr: minimum learning rate
        mode: "cosine_warmup" or "cosine_warm_restart"
        restart_period: steps per restart cycle (for warm restart mode)
    """
    warmup_steps = int(total_steps * warmup_ratio)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    if mode == "cosine_warm_restart" and restart_period > 0:
        def lr_lambda_factory(base_lr):
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                # SGDR: cosine annealing with warm restarts
                t = step - warmup_steps
                cycle_pos = t % restart_period
                progress = float(cycle_pos) / float(restart_period)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return max(min_lr / base_lr, cosine)
            return lr_lambda
    else:
        def lr_lambda_factory(base_lr):
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return max(min_lr / base_lr, cosine)
            return lr_lambda

    lambdas = [lr_lambda_factory(lr) for lr in base_lrs]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return scheduler
