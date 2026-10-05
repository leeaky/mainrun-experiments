# 02 — Optimizer: SGD → AdamW

> **Status:** kept · **Effect:** val loss 1.75193 → 1.24794 (**Δ −0.504**) · **Date:** 2026-09-07
>
> ← [01-baseline-reproduction.md](01-baseline-reproduction.md) · [timeline](README.md) · [next](03-scaled-residual-init.md) →

## Question
What is the effect of replacing un-momentumized SGD with decoupled AdamW, and what is the optimal peak learning rate under a warmup-cosine schedule?

## Reasoning
SGD with a scalar learning rate treats all dimensions identically. In transformers, gradients across embeddings, attention projections, and feed-forward layers span several orders of magnitude. Furthermore, 7 epochs (~1,000 steps) is an extremely constrained optimization horizon where second-moment normalization and adaptive step sizing are essential for rapid convergence.

## Approach
1. Implemented standard two-way parameter splitting in `train.py`:
   - 2D weight matrices (Linear weights) receive weight decay (`weight_decay=0.01`).
   - 1D tensors (LayerNorm weights and biases, embedding biases) receive no weight decay (`weight_decay=0.0`).
2. Configured warm-up cosine annealing: linear ramp for 6% of total steps (`warmup_frac=0.06`), decaying to 10% of peak LR (`lr_floor_frac=0.10`).
3. Swept peak learning rate across a bracket: `[4e-4, 8e-4, 1.2e-3, 1.6e-3, 2.0e-3, 3.0e-3]`.

## Results

### Learning Rate Bracket (AdamW, `wd=0.01`, `warmup=0.06`, `cosine`)
| Learning Rate (`lr`) | Best Val Loss | Step | Final Val Loss | Notes |
|:---:|:---:|:---:|:---:|---|
| 4e-4 | 1.34863 | 1001 | 1.34863 | Under-training |
| 8e-4 | 1.27218 | 1001 | 1.27218 | Monotonic improvement |
| **1.2e-3** | **1.24794** | 1001 | **1.24794** | **Optimal peak LR** |
| 1.6e-3 | 1.25201 | 1001 | 1.25201 | Mild overshooting |
| 2.0e-3 | 1.27341 | 1001 | 1.27341 | Instability begins |
| 3.0e-3 | 1.36502 | 1001 | 1.36502 | Severe degradation |

Comparison against baseline:
- Baseline (SGD, `lr=6e-3`): **1.75193**
- AdamW (`lr=1.2e-3`, `wd=0.01`): **1.24794**
- **Net improvement: Δ −0.50399**

## What we learned
- The optimizer change is overwhelmingly the dominant lever of the entire project, delivering over 85% of total obtainable gains.
- The optimal learning rate for this 6-layer GPT-2 model under 1,000 steps is `1.2e-3`.
- The loss curve shows dramatic differences in early dynamics: by step 143, AdamW reaches val loss 1.543, whereas SGD took over 900 steps to reach that point.

## Open threads
- Test architectural stabilization via GPT-2 style scaled residual initialization ([03-scaled-residual-init.md](03-scaled-residual-init.md)).

## Sources
- Code: [`mainrun/train.py`](../../mainrun/train.py)
- Source notes: `docs/_source/notes/2026-09-07-exp-4.1-adamw.md`, `docs/_source/notes/2026-09-07-adamw-lr-bracket.md`
