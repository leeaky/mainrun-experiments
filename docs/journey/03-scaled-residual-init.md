# 03 — Scaled Residual Initialization

> **Status:** kept under AdamW · **Effect:** val loss 1.24794 → 1.20428 (**Δ −0.044**) · **Date:** 2026-09-07
>
> ← [02-adamw.md](02-adamw.md) · [timeline](README.md) · [next](04-config-sweeps.md) →

## Question
Does scaling the initial weights of the residual output projections by $1/\sqrt{2 N_{\text{layer}}}$ improve training dynamics and final validation loss?

## Reasoning
In deep residual networks, activations accumulate variance as they pass through successive residual additions:
$$x_{l+1} = x_l + f(x_l)$$
Without scaling, the variance grows linearly with depth $L$. Following GPT-2 / Radford et al., scaling the projection matrices that feed back into the residual stream (`c_proj` in attention and `w_down` in MLP) by $\sigma = 0.02 / \sqrt{2 N_{\text{layer}}}$ preserves the identity mapping early in training, allowing gradients to propagate cleanly through the skip connections.

## Approach
Implemented the scaled initialization in `GPT._init_weights`:
```python
if cfg.scaled_residual_init:
    std = 0.02 / math.sqrt(2 * cfg.n_layer)
    for block in self.blocks:
        nn.init.normal_(block.attn.proj.weight, mean=0.0, std=std)
        mlp_out = block.mlp.net[2] if hasattr(block.mlp, "net") else block.mlp.w_down
        nn.init.normal_(mlp_out.weight, mean=0.0, std=std)
```
Tested across both optimizers (SGD and AdamW) to evaluate interaction effects.

## Results

| Optimizer Configuration | Standard Init | Scaled Residual Init | Δ |
|---|:---:|:---:|:---:|
| Baseline (SGD, `lr=6e-3`) | 1.75193 | 1.69777 | **−0.05416** |
| AdamW (`lr=1.2e-3`, `wd=0.01`) | 1.24794 | **1.20428** | **−0.04366** |

The combo recipe at this stage reached **1.20428** (an overall drop of **−0.548** from the original baseline).

## What we learned
- Scaled residual initialization is an unambiguous win under both SGD and AdamW, providing a consistent ~0.04–0.05 reduction in validation loss.
- It prevents early saturation of activations in the upper layers, accelerating learning throughout all 7 epochs.
- *(Foreshadowing: Later in Phase 10/11, under the Muon optimizer, this effect reversed slightly (+0.002) because Newton-Schulz orthogonalization normalizes update scales independently of parameter initialization. But under AdamW, it is essential).*

## Open threads
- Run systematic sweeps across non-architectural configuration fields: `block_size`, `batch_size`, `dropout`, `vocab_size`, and model capacity ([04-config-sweeps.md](04-config-sweeps.md)).

## Sources
- Code: [`mainrun/train.py`](../../mainrun/train.py)
- Source note: `docs/_source/notes/2026-09-07-exp-4.5-scaled-residual-init.md`
