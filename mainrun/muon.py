"""Muon optimizer (Momentum Orthogonalized by Newton-Schulz) + a small
combined-optimizer shim. Ported from Keller Jordan's modded-nanogpt.

Muon orthogonalizes each 2D parameter's momentum before applying it, so the
update uses the matrix's full spectrum each step. Use it ONLY for hidden
weight matrices; keep embeddings / the tied head / norms / biases on AdamW.
"""
import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration -> approx. orthogonal factor of G.

    fp32 for run-to-run determinism (bf16 is faster but not bit-exact, which
    breaks the 0.0 noise-floor workflow this repo relies on).
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(list(params),
                         dict(lr=lr, momentum=momentum, nesterov=nesterov,
                              ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nesterov else buf
                g = zeropower_via_newtonschulz5(g, steps=ns)
                # RMS of an orthogonal update is ~1; rescale wide/tall matrices
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(g.type_as(p), alpha=-group["lr"] * scale)


class CombinedOptimizer:
    """Steps several sub-optimizers together and applies ONE multiplicative
    schedule factor to every param group. Not a torch.optim.Optimizer - just
    the surface train.py's loop touches (zero_grad / step / param_groups).
    """

    def __init__(self, optimizers, sched_fn):
        self.optimizers = list(optimizers)
        self.sched_fn = sched_fn
        self._base = [[g["lr"] for g in o.param_groups] for o in self.optimizers]
        self.apply_schedule(0)

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def apply_schedule(self, step0: int):
        f = self.sched_fn(step0)
        for o, bases in zip(self.optimizers, self._base):
            for g, base in zip(o.param_groups, bases):
                g["lr"] = base * f

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]
