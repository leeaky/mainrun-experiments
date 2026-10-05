"""Exponential moving average of model weights, evaluated instead of the
live weights. Pure eval-time transform - training dynamics are untouched, so
this never changes step count / data / RNG. decay <= 0 disables everything.

Plain constant decay, no warmup ramp: the runs here are ~1000 steps, far
too short for the classic `(1+step)/(k+step)` ramp to saturate (it would
just clamp every decay to ~0.99). The shadow starts equal to the init
weights, so the first few evals are init-dominated and high - but
`best_val_loss` takes the min and `final_val_loss` is at the last step,
where for any usable decay (<= ~0.995 at this run length) the init
contribution has washed out. Decays above ~0.995 lag too much to be useful
here regardless.
"""
from contextlib import contextmanager

import torch


class EMA:
    def __init__(self, named_params, decay: float):
        self.decay = float(decay)
        self.shadow = {}
        if self.decay > 0:
            for n, p in named_params:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, named_params, step: int):
        if self.decay <= 0:
            return
        d = self.decay
        for n, p in named_params:
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextmanager
    def swapped_in(self, model):
        if self.decay <= 0:
            yield
            return
        backup = {}
        for n, p in model.named_parameters():
            backup[n] = p.detach().clone()
            p.data.copy_(self.shadow[n])
        try:
            yield
        finally:
            for n, p in model.named_parameters():
                p.data.copy_(backup[n])
